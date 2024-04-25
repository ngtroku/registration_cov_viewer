// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

#include <iostream>
#include <small_gicp/benchmark/read_points.hpp>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include <guik/viewer/light_viewer.hpp>

using namespace small_gicp;

// ２つの点群間のICPの誤差解析
void example1(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 4;                       // Number of threads to be used
  double downsampling_resolution = 0.25;     // Downsampling resolution
  int num_neighbors = 10;                    // Number of neighbor points used for normal and covariance estimation
  double max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

  auto viewer = guik::viewer();

  /***  A. 点群レジストレーションとヘッセ行列の取得  ***/
  // 点群の前処理
  auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
  auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

  // 点群レジストレーション
  RegistrationSetting setting;
  setting.type = RegistrationSetting::GICP;
  setting.num_threads = num_threads;
  setting.max_correspondence_distance = max_correspondence_distance;

  Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
  auto result = align(*target, *source, *target_tree, init_T_target_source, setting);

  viewer->update_points("target", target->points, guik::FlatBlue());
  viewer->update_points("source", source->points, guik::FlatRed(result.T_target_source));

  // 推定結果 (相対姿勢＆目的関数のヘッセ行列)，ヘッセ行列は並進・回転 [rx, ry, rz, tx, ty, tz] の線形空間での二階微分
  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "--- Hessian ---" << std::endl << result.H << std::endl;

  // ヘッセ行列の逆行列は共分散行列の近似になる
  // 共分散行列もヘッセ行列と同様に線形空間 [rx, ry, rz, tx, ty, tz] で表される
  Eigen::Matrix<double, 6, 6> cov = result.H.inverse();
  std::cout << "--- cov ---" << std::endl << cov << std::endl;

  // 共分散を表す球を表示する
  Eigen::Affine3d sphere_matrix = Eigen::Affine3d::Identity();
  sphere_matrix.linear() = cov.block<3, 3>(3, 3) * 5e5;  // 共分散が小さいので大きく表示するためにスケーリング
  sphere_matrix.translation() = result.T_target_source.translation();

  viewer->update_wire_sphere("cov", guik::FlatColor(1.0, 0.0, 0.0, 1.0, sphere_matrix));

  /***  B. 各点のヘッセ行列から全体ヘッセ行列を計算する  ***/
  std::vector<GICPFactor> factors(source->size());  // 各点ごとにICPコスト計算クラスを作成
  DistanceRejector rejector;                        // 一定距離以上の対応付は棄却する
  rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

  // 各点のヘッセ行列
  std::vector<Eigen::Matrix<double, 6, 6>> Hs(source->size(), Eigen::Matrix<double, 6, 6>::Zero());
  for (size_t i = 0; i < source->size(); i++) {
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    double e;
    if (factors[i].linearize(*target, *source, *target_tree, result.T_target_source, i, rejector, &H, &b, &e)) {
      // 計算されたヘッセ行列を保存
      Hs[i] = H;
    } else {
      // 対応点が一定距離以上で棄却されたら場合は最終結果に影響しない
    }
  }

  // 各点のヘッセ行列を合計したものは，最初に求めたヘッセ行列とほぼ一致する
  Eigen::Matrix<double, 6, 6> sum_Hs = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& H : Hs) {
    sum_Hs += H;
  }

  std::cout << "--- sum_Hs ---" << std::endl << sum_Hs << std::endl;

  /*** C. ヘッセ行列の解析 ***/
  // 固有値分解によって拘束の強さを調べる
  // 固有値の大きさ＝拘束の強さを表し，対応する固有ベクトル＝拘束の方向を表す
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig_H(sum_Hs);
  std::cout << "--- Eigenvalues ---" << std::endl << eig_H.eigenvalues().transpose() << std::endl;
  std::cout << "--- Eigenvectors ---" << std::endl << eig_H.eigenvectors() << std::endl;

  // 最小固有値は最も拘束の弱い方向を表し，対応する固有ベクトルが最も拘束の弱い方向を表す
  std::cout << "minimum eigenvalue: " << eig_H.eigenvalues()[0] << std::endl;
  Eigen::Matrix<double, 6, 1> min_eigenvector = eig_H.eigenvectors().col(0);  // [rx, ry, rz, tx, ty, tz]

  // 最も拘束が弱い方向(=最も共分散が大きい方向)を表示する
  Eigen::Vector3d center = result.T_target_source.translation();
  std::vector<Eigen::Vector3d> line = {center, center + min_eigenvector.tail<3>() * 2.0};
  viewer->update_thin_lines("min_eigenvector", line, false, guik::FlatGreen());

  viewer->spin();
}

// １つの点群における自分自身とのICPの誤差解析
void example2(const std::vector<Eigen::Vector4f>& target_points) {
  auto target = std::make_shared<PointCloud>(target_points);
  target = voxelgrid_sampling(*target, 0.1);

  // まったく同一の点群だと誤差=0で計算ができないため，ランダムサンプリング＋ノイズ付加して擬似的に２つの点群に分ける
  std::mt19937 mt;
  std::normal_distribution<> ndist(0.0, 0.01);

  auto target_A = random_sampling(*target, 5000, mt);
  std::for_each(target_A->points.begin(), target_A->points.end(), [&](Eigen::Vector4d& p) { p += Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0); });
  estimate_covariances(*target_A, 10);
  auto target_A_tree = std::make_shared<KdTree<PointCloud>>(target_A);

  auto target_B = random_sampling(*target, 5000, mt);
  std::for_each(target_B->points.begin(), target_B->points.end(), [&](Eigen::Vector4d& p) { p += Eigen::Vector4d(ndist(mt), ndist(mt), ndist(mt), 0.0); });
  estimate_covariances(*target_B, 10);

  /***  B. 各点のヘッセ行列から全体ヘッセ行列を計算する  ***/
  std::vector<GICPFactor> factors(target_B->size());  // 各点ごとにICPコスト計算クラスを作成
  DistanceRejector rejector;                          // 一定距離以上の対応付は棄却する
  rejector.max_dist_sq = 1.0;

  // 各点のヘッセ行列
  std::vector<Eigen::Matrix<double, 6, 6>> Hs(target_B->size(), Eigen::Matrix<double, 6, 6>::Zero());
  for (size_t i = 0; i < target_B->size(); i++) {
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    double e;
    if (factors[i].linearize(*target_A, *target_B, *target_A_tree, Eigen::Isometry3d::Identity(), i, rejector, &H, &b, &e)) {
      // 計算されたヘッセ行列を保存
      Hs[i] = H;
    } else {
      // 対応点が一定距離以上で棄却されたら場合は最終結果に影響しない
    }
  }

  Eigen::Matrix<double, 6, 6> sum_Hs = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& H : Hs) {
    sum_Hs += H;
  }
  std::cout << "--- sum_Hs ---" << std::endl << sum_Hs << std::endl;

  Eigen::Matrix<double, 6, 6> cov = sum_Hs.inverse();
  std::cout << "--- cov ---" << std::endl << cov << std::endl;

  // 共分散を表す球を表示する
  Eigen::Affine3d sphere_matrix = Eigen::Affine3d::Identity();
  sphere_matrix.linear() = cov.block<3, 3>(3, 3) * 5e5;  // 共分散が小さいので大きく表示するためにスケーリング
  sphere_matrix.translation() = Eigen::Vector3d::Zero();

  // 固有値分解によって拘束の強さを調べる
  // 固有値の大きさ＝拘束の強さを表し，対応する固有ベクトル＝拘束の方向を表す
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig_H(sum_Hs);
  std::cout << "--- Eigenvalues ---" << std::endl << eig_H.eigenvalues().transpose() << std::endl;
  std::cout << "--- Eigenvectors ---" << std::endl << eig_H.eigenvectors() << std::endl;

  // 最小固有値は最も拘束の弱い方向を表し，対応する固有ベクトルが最も拘束の弱い方向を表す
  std::cout << "minimum eigenvalue: " << eig_H.eigenvalues()[0] << std::endl;
  Eigen::Matrix<double, 6, 1> min_eigenvector = eig_H.eigenvectors().col(0);  // [rx, ry, rz, tx, ty, tz]

  // 最も拘束が弱い方向(=最も共分散が大きい方向)を表示する
  std::vector<Eigen::Vector3d> line = {Eigen::Vector3d::Zero(), min_eigenvector.tail<3>() * 2.0};

  auto viewer = guik::viewer();
  viewer->update_points("target_A", target_A->points, guik::FlatBlue());
  viewer->update_points("target_B", target_B->points, guik::FlatRed());
  viewer->update_wire_sphere("cov", guik::FlatColor(1.0, 0.0, 0.0, 1.0, sphere_matrix));
  viewer->update_thin_lines("min_eigenvector", line, false, guik::FlatGreen());
  viewer->spin();
}

int main(int argc, char** argv) {
  std::vector<Eigen::Vector4f> target_points = read_ply("data/target_cropped.ply");
  std::vector<Eigen::Vector4f> source_points = read_ply("data/source.ply");
  if (target_points.empty() || source_points.empty()) {
    std::cerr << "error: failed to read points from data/(target|source).ply" << std::endl;
    return 1;
  }

  example1(target_points, source_points);
  example2(target_points);

  return 0;
}