# registration_cov_viewer
点群レジストレーションにおいて重要な点を探索し、可視化する。

# Dependencies
small_gicp https://github.com/koide3/iridescence

Other python libraries can be downloaded `requirements.txt`
`pip install -r ./requirements.txt`

# How to use
## gicp_matching.py (点群レジストレーション)
1. 読み込みたい点群ファイルのpathを入力
   
   読み込むファイルが.csv形式の場合、読み込みフォーマットを選択（VLP-16 or AT128 or pcd）
   
   サンプルファイルの場合はAT128
   
   example:`Please set data file(.csv or .bin) >>  ./data/000000.bin`
   
2. ランダムにノイズを与えてシミュレーションする回数を選択

   １以上の整数を入力

   example:`How many times do you simulate matching? >> 5`

3. 結果を保存するpathを入力
   
   .csv形式で保存する

   example:`Please set save path (.csv) >> ./results/result_000.csv`

## visualizer.py (結果の可視化)
読み込みたい結果ファイルのpathを入力（.csv形式）

点の値は0〜1をとる。0に近いほど青く表示され、1に近いほど赤く表示される。
