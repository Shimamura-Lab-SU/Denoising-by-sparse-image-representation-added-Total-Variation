
%	Transform k-SVD を用いた辞書学習とOMP(orADMM)を用いた画像復元

clear

close all

% --- パラメータ群 ---

%	* 画像の設定
ImNames={'mandril'};				% (画像名).png，複数指定すると画像ごとに雑音除去を行う．
% ImNames={'lena','barbara','boats','house','peppers256'};	% 画像を複数指定した例
% ImNames={'cameraman', 'house', 'jetplane', 'lake', 'lena', 'livingroom', 'mandril', 'peppers', 'pirate', 'walkbridge', 'woman_blonde', 'woman_darkhair'};	% 画像を複数指定した例

SigmaVec=[40];					% 雑音分散，複数指定するとそれごとに雑音除去を行う．
% SigmaVec=[5, 10, 15, 20, 35, 30, 35, 40];		% 雑音分散を複数指定した例

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
	imshow(I,[0,255]);

	X	= im2col(I,[n,n],'sliding');					% ミニバッチ(n×n)ごとに画像をスライス，列ベクトルで並べる
	Xdn	= zeros(size(X));

	disp(['Image = ',ImNames{j}])

	for k=1:length(SigmaVec)

		% --- 雑音重畳画像の生成 ---

		%	* 雑音の生成と加算
		sigma	= SigmaVec(k);							% SigmaVec に収まっている雑音分散ごとに画像生成
        disp(['sigma = ', num2str(SigmaVec)]);
% 		In		= I + randn(size(I))*sigma;				% 画像にガウス性白色雑音を重畳
        In       = importdata(['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Noiseimg_σ=', num2str(sigma), '_', ImNames{j}, '.txt']);
        
		%	* 2次元画像 -> ミニバッチデータ
		Xn		= im2col(In,[n,n],'sliding');			% ミニバッチ(n×n)ごとに雑音重畳画像をスライス，列ベクトルで並べる

		%	* PSNRの計算
		PSNRin	= 10*log10(255.^2/mean((In(:)-I(:)).^2));	% 初期のPSNR
		figure,imshow(In,[0,255])
		title(['Noise Image : PSNR = ',num2str(PSNRin),' dB'])
		disp(['Initial PSNR [dB] : ',num2str(PSNRin)]);	% 表示
		

		% --- 初期設定 ---
		Xdn		= zeros(size(Xn));						% 空行列
		Std0	= 1.15*sigma;							% 平坦(<Std0)部分の検出閾値
		Std1	= 1.5*sigma;							% テクスチャ(>Std1)部分の検出閾値
		cnt		= countcover([256 256],[n n],[1 1]);	% オーバーラップ回数の測定行列

        
        
		% --- 辞書学習 ---
        
		% 画像を平坦部分とテクスチャ(構造)部分にわけて，テクスチャ部分のみから基底を学習(?)

		%	* 平坦とテクスチャを分割

		%	 - 平坦部分の分離と処理
		Id0			= find(std(Xn)<Std0);   			% 分散が Std0 以下は平坦部分とみなす 平坦部分を1とする
		Xdn(:,Id0)	= repmat(mean(Xn(:,Id0)),d,1);		% 平坦部分を平均化処理でならす   %mean(Xn(:,Id0))を要素に持つd行1列の行列を生成

		%	 - テクスチャ部分の分離
		IdxT		= find(std(Xn)>=Std1);				% 分散が Std1 以上でテクスチャ(構造)とみなす & そのインデックス番号取得
		IdxN		= randperm(min(N, length(IdxT)));	% テクスチャのインデックス番号をランダムで入れ替え & N個のテクスチャを抜き出す(N>テクスチャ数, ならすべてのテクスチャを抜き出す)
		Xntrain		= Xn(:,IdxN);						% データからテクスチャを抜き出す


		%	* テクスチャから辞書学習 -> 辞書：Omega
% 		Omega		= TransformKSVD(Xntrain,DimTar,p,length(IdxN),IterT,qq);
        Omega       = importdata(['./dataset/', ImNames{j}, '/σ=', num2str(sigma), '/Dic_noise_σ=', num2str(sigma), '_', ImNames{j},'.txt']);

		%	* 辞書の図示
		dic			= figure(1);
 		DisplayOmega(Omega, dic);						% 学習した辞書を表示
		title('Dictionary')
		disp(['Number of examples: ',num2str(size(Xntrain,2))]);


		% --- 画像復元 ---
		% 画像の完全に平坦な部分(直流)とそれ以外にわけて，完全に平坦な部分以外を復元(?)

		%	* 画像データ整形
		Idx			= find(std(Xn)>=Std0);				% 分散が Std0 以下は直流のみとみなして処理しない インデックスを保存
		Xna			= Xn(:,Idx);						% 直流のみ以外のミニバッチデータを抽出

		%	* ミニバッチデータから直流成分(平均値)を除去
		DCn			= mean(Xna);						% ミニバッチごとの平均を算出
		Xna			= Xna - repmat(DCn,d,1);			% 平均値を差し引いて直流成分を除去


% =========================================================================
%                            ADMM による画像復元
% =========================================================================
    
        T = zeros([1,1]);
        P = zeros([1,1]);
        S = zeros([1,1]);
        count = 1;
        
        for theta=0:0.1:1
       
            disp(['theta = ', num2str(theta)]);

            %	* ADMMの計算に必要な行列の計算

            Phy_coef	= eye(n*n);     % 係数のスパース性に関する行列
            
%-----------差分計算用行列の作成--------------------------------------------

% % ----------横方向---------------------------------------------------------
% 
%             Phy_tv		= eye(n*n) - circshift(eye(n*n),[-1,0]);	% 差分計算用行列
%             
%-----------横方向 + 縦方向-------------------------------------------------
%-----------横方向---------------------------------------------------------
            a1 = repmat(1,1,n*n);

            for i=1:1:n
                a1(:, n*i) = 0;
            end

            A1 = diag(-a1, 1);      %横方向
            B1  = diag(a1);         %対角
            
            Phy_tv1 = (A1(1:n*n,1:n*n) + B1(1:n*n,1:n*n)) * Omega;     %   横方向
            
            

%             disp(Phy_tv1);
%             dlmwrite('TV1.txt', Phy_tv1); 
%--------------------------------------------------------------------------       
%-----------縦方向---------------------------------------------------------

            a2 = repmat(1,1,n*n);
            
            A2 = diag(-a2, n);      %縦方向
            B2  = diag(a2);         %対角

            A2(n*n-n+1:n*n,:) = 0;
            B2(n*n-n+1:n*n,:) = 0;
            
            Phy_tv2 = (A2(1:n*n,1:n*n) + B2(1:n*n,1:n*n)) * Omega;     %   縦方向
            
%             disp(Phy_tv2);
%             dlmwrite('TV2.txt', Phy_tv2); 

%--------------------------------------------------------------------------

            Phy_tv  = [Phy_tv1; Phy_tv2];
            
% %             %   外側
% %             a   = repmat(-0.25,1,n*n);
% %             
% %             %   内側
% %             b   = repmat(-0.25,1,n*n);
% %             
% %             for i=1:1:n*n
% %                 b(:, n*i) = 0;
% %             end
% %             
% %             %   対角
% %             D   = eye(n*n);
% %             
% %             %   対角行列化
% %             A1 = diag(a,n);
% %             A2 = diag(a,-n);
% %             B1 = diag(b,1);
% %             B2 = diag(b,-1);
% % 
% %             %   合成してTV正則化項に
% %             Phy_tv  = A1(1:n*n,1:n*n) + A2(1:n*n,1:n*n) + B1(1:n*n,1:n*n) + B2(1:n*n,1:n*n) + D;	% 差分計算用行列
% %             
% 
% %             disp(Phy_tv);
%             
% %--------------------------------------------------------------------------
            %	- 正則化行列 (注意！たぶん正しくない．自分で変えて)
            Dr = [Phy_coef; theta .* Phy_tv];

            
            %	* ADMMによる復元
            lambda	= 6.7;

            w		= admm(Omega, Xna, Dr, lambda);		% ADMMによりスパース1係数を取得
            Xdna	= Omega * w;						% 画像復元

            %	* ミニバッチデータに直流成分(平均値)を加算
            Xdna	= Xdna + repmat(DCn,d,1);				% 直流成分を元にもどす
            Xdn(:,Idx) = Xdna;

            %	* ミニバッチデータを元の２次元画像データに変換
            Idn		= col2imstep(Xdn,[256 256],[n n])./cnt;

            % --- 復元結果 ---
            %	* PSNRの計算
            PSNRout = 10*log10(255.^2/mean((Idn(:)-I(:)).^2));
            disp(['Output PSNR [dB] : ',num2str(PSNRout)]);
            
            %	* SSIMの計算
            SSIM = ssim(Idn, I);
            disp(['Output SSIM : ',num2str(SSIM)]);

            %   * 復元画像の表示
            figure,imshow(uint8(Idn),[0,255])	
            title(['Restroration Image : PSNR = ',num2str(PSNRout),' dB'])
            
            %   * 復元画像の保存
            rootpath = ['./result_image/proposal/theta/result/', ImNames{j},'/σ=', num2str(sigma), '/'];   %保存ファイルパス

%             restored_img = [rootpath,'theta=',num2str(theta), '.png']; % ファイル名の作成
            restored_img = [rootpath,'SSIM_theta=',num2str(theta), '.png']; % SSIM

            saveas(gcf,restored_img) % ファイルへの保存


            T(:,count) = theta;
            P(:,count) = PSNRout;
            S(:,count) = SSIM;
            
            count = count + 1;
            
        end
        
%         disp(T);
%         disp(P);
%         disp(S);

   
    
%-----------------------------PSNRの結果出力-------------------------------
%     %    * PSNRグラフの表示,保存
%     path_graph_PSNR = [rootpath, 'PSNR_graph.png'];
%     plot(T,P);
%     saveas(gcf,path_graph_PSNR) % ファイルへの保存
% 
%     %    * PSNRの最大値とλの値
%     disp(['Image = ',ImNames{j}, '   sigma = ', num2str(sigma)]);
%     [bestPSNR, index_PSNR] = max(P);
%     disp(['bestPSNR = ', num2str(bestPSNR),'   theta  =', num2str(T(:,index_PSNR))]);
% 
%     %    * PSNRの結果をファイルに出力
%     A_PSNR = [T; P];
%     fileID = fopen([rootpath, 'result_PSNR.txt'],'w');
%     fprintf(fileID,'\n theta = %4.1f   bestPSNR = %7.5f\n',T(:,index_PSNR), bestPSNR);
%     fprintf(fileID,' 経過時間 : %f\n\n', toc);
%     fprintf(fileID,'%9s 　%9s\n\n', 'theta', 'PSNR');
%     fprintf(fileID,'%9.1f 　%9.5f\n', A_PSNR);
%     fclose(fileID);
%     
    
    %------------------------------SSIMの結果出力------------------------------

%        * SSIMグラフの表示,保存
    path_graph_SSIM = [rootpath, 'SSIM_graph.png'];
    plot(T,S);
    saveas(gcf,path_graph_SSIM) % ファイルへの保存
    
%        * SSIMの最大値とλの値
	disp(['Image = ',ImNames{j}])
    disp(['sigma = ', num2str(sigma)]);
    [bestSSIM, index_SSIM] = max(S);
    disp(['bestSSIM = ', num2str(bestSSIM),'   theta  =', num2str(T(:,index_SSIM))]);

%        * SSIMの結果をファイルに出力
    A_SSIM = [T; S];
    fileID = fopen([rootpath, 'result_SSIM.txt'],'w');
    fprintf(fileID,' theta = %4.1f   bestSSIM = %7.5f\n',T(:,index_SSIM), bestSSIM);
    fprintf(fileID,' 経過時間 : %f\n\n', toc);
    fprintf(fileID,'%9s 　%9s\n\n', 'theta', 'SSIM');
    fprintf(fileID,'%9.1f 　%9.5f\n', A_SSIM);
    fclose(fileID);

    clear bestPSNR;
    clear bestSSIM;
    clear index;

    end
    
end

toc;

