
%	Transform k-SVD を用いた辞書学習とOMP(orADMM)を用いた画像復元

clear

close all

% --- パラメータ群 ---

%	* 画像の設定
% ImNames={'house'};				% (画像名).png，複数指定すると画像ごとに雑音除去を行う．
ImNames={'cameraman', 'house', 'jetplane', 'lake', 'lena', 'livingroom','mandril','peppers', 'pirate', 'walkbridge', 'woman_blonde', 'woman_darkhair'};	% 画像を複数指定した例

% SigmaVec=[15];					% 雑音分散，複数指定するとそれごとに雑音除去を行う．
% SigmaVec=[5, 10, 15, 20, 25, 30];		% 雑音分散を複数指定した例
SigmaVec=[35, 40];		% 雑音分散を複数指定した例

%	* 任意のパラメータ
n		= 7;					% ミニバッチの縦・横の大きさ
N		= 20000;				% 学習用データ数
p		= n*n;					% 辞書のアトム数(基底数)
IterT	= 50;					% k-SVDの学習回数
qq		= 4;					% 復元の際に用いる基底数

%	* 一意に決まるパラメータ (任意のパラメータから自動で決まる)
d		= n^2;					% ミニバッチのピクセル数
DimTar	= n;					% Tramsform k-SVD の基底サイズ

tic;


for j=1:length(ImNames)

	%   I           :   原画像(256,256)
	%   In          :   Iにノイズを加えたもの
	%   X           :   I をミニバッチごとに縦に並べたもの
	%   Xn          :   Inをミニバッチごとに縦に並べたもの
	%   Xdn         :   Xnの平坦部分を平均化したもの
	%   Id0         :   Xnの平坦部分を1としたもの
	%   IdxT        :   Xnのテクスチャ部分を1としたもの
	%   Xna         :   Xnのテクスチャ部分を抜き出したものから直流成分を取り除いたもの
	%   DCn         :   Xnaの直流成分
	%   Xntrain     :   Xnのテクスチャ部分を平均化したもの
	%   Xntrain_r   :   Xnをランダムに並び替えたもの
    
	% --- 画像の読み込み ---
% 	I	= double(imread(['./images/', ImNames{j},'.png']));	% 読み込みとdouble型変換
    t = Tiff(['./Standard Image Database/', ImNames{j},'.tif'],'r');
    I = read(t);   
    I= I(:,:,1);
    I = imresize(I, [256, 256]);
    I = double(I);
    imshow(uint8(I));
    
	X	= im2col(I,[n,n],'sliding');					% ミニバッチ(n×n)ごとに画像をスライス，列ベクトルで並べる
	Xdn	= zeros(size(X));

	disp(['Image = ',ImNames{j}]);

	for k=1:length(SigmaVec)

		% --- 雑音重畳画像の生成 ---

		%	* 雑音の生成と加算
		sigma	= SigmaVec(k);							% SigmaVec に収まっている雑音分散ごとに画像生成
        disp(['sigma = ', num2str(sigma)]);
        In		= I + randn(size(I))*sigma;				% 画像にガウス性白色雑音を重畳
        dlmwrite(['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Noiseimg_σ=', num2str(sigma), '_', ImNames{j}, '.txt'], In);                   %雑音画像を保存(txt形式)

		%	* 2次元画像 -> ミニバッチデータ
		Xn		= im2col(In,[n,n],'sliding');			% ミニバッチ(n×n)ごとに雑音重畳画像をスライス，列ベクトルで並べる       

		%	* PSNRの計算
		PSNRin	= 10*log10(255.^2/mean((In(:)-I(:)).^2));	% 初期のPSNR
        
		figure,imshow(In,[0,255])
		title(['Noise Image : PSNR = ',num2str(PSNRin),' dB'])
		disp(['Initial PSNR [dB] : ',num2str(PSNRin)]);	% 表示
        saveas(gcf, ['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Noiseimg_σ=', num2str(sigma), '_', ImNames{j}, '.png']);                     %雑音画像を保存(png形式)

		% --- 初期設定 ---
		Xdn		= zeros(size(Xn));						% 空行列
		Std0	= 1.15*sigma;							% 平坦(<Std0)部分の検出閾値
		Std1	= 1.5*sigma;							% テクスチャ(>Std1)部分の検出閾値
		cnt		= countcover([256 256],[n n],[1 1]);	% オーバーラップ回数の測定行列
        
        
		% --- 辞書学習 ---
        
		% 画像を平坦部分とテクスチャ(構造)部分にわけて，テクスチャ部分のみから基底を学習(?)

		%	* 平坦とテクスチャを分割

        %   雑音画像から辞書を生成
		%	 - 平坦部分の分離と処理
		Id0			= find(std(Xn)<Std0);   			% 分散が Std0 以下は平坦部分とみなす 平坦部分を1とする
		Xdn(:,Id0)	= repmat(mean(Xn(:,Id0)),d,1);		% 平坦部分を平均化処理でならす   %mean(Xn(:,Id0))を要素に持つd行1列の行列を生成

		%	 - テクスチャ部分の分離
		IdxT		= find(std(Xn)>=Std1);				% 分散が Std1 以上でテクスチャ(構造)とみなす & そのインデックス番号取得
		IdxN		= randperm(min(N, length(IdxT)));	% テクスチャのインデックス番号をランダムで入れ替え & N個のテクスチャを抜き出す(N>テクスチャ数, ならすべてのテクスチャを抜き出す)
		Xntrain		= Xn(:,IdxN);						% データからテクスチャを抜き出す

%         %   原画像から辞書を生成
%         %	 - 平坦部分の分離と処理
%         Id0			= find(std(X)<Std0);   			% 分散が Std0 以下は平坦部分とみなす 平坦部分を1とする
% 		Xdn(:,Id0)	= repmat(mean(X(:,Id0)),d,1);		% 平坦部分を平均化処理でならす   %mean(Xn(:,Id0))を要素に持つd行1列の行列を生成
% 
%         %	 - テクスチャ部分の分離
% 		IdxT		= find(std(X)>=Std1);				% 分散が Std1 以上でテクスチャ(構造)とみなす & そのインデックス番号取得
% 		IdxN		= randperm(min(N, length(IdxT)));	% テクスチャのインデックス番号をランダムで入れ替え & N個のテクスチャを抜き出す(N>テクスチャ数, ならすべてのテクスチャを抜き出す)
% 		Xntrain		= X(:,IdxN);						% データからテクスチャを抜き出す


		%	* テクスチャから辞書学習 -> 辞書：Omega
		Omega		= TransformKSVD(Xntrain,DimTar,p,length(IdxN),IterT,qq);
        dlmwrite(['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Dic_noise_σ=', num2str(sigma), '_', ImNames{j},'.txt'], Omega);                     %雑音画像から作った辞書の保存

        
		%	* 辞書の図示
		dic			= figure(1);
 		DisplayOmega(Omega, dic);						% 学習した辞書を表示
		title('Dictionary')
		disp(['Number of examples: ',num2str(size(Xntrain,2))]);
        saveas(gcf, ['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Dic_noise_σ=', num2str(sigma), '_', ImNames{j},'.png']);                     %雑音画像を保存(png形式)
       
    end
    
end


toc;

