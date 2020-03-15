
%	Transform k-SVD ��p���������w�K��OMP(orADMM)��p�����摜����

clear

close all

% --- �p�����[�^�Q ---

%	* �摜�̐ݒ�
ImNames={'mandril'};				% (�摜��).png�C�����w�肷��Ɖ摜���ƂɎG���������s���D
% ImNames={'lena','barbara','boats','house','peppers256'};	% �摜�𕡐��w�肵����
% ImNames={'cameraman', 'house', 'jetplane', 'lake', 'lena', 'livingroom', 'mandril', 'peppers', 'pirate', 'walkbridge', 'woman_blonde', 'woman_darkhair'};	% �摜�𕡐��w�肵����

SigmaVec=[40];					% �G�����U�C�����w�肷��Ƃ��ꂲ�ƂɎG���������s���D
% SigmaVec=[5, 10, 15, 20, 35, 30, 35, 40];		% �G�����U�𕡐��w�肵����

%	* �C�ӂ̃p�����[�^
n		= 7;					% �~�j�o�b�`�̏c�E���̑傫��
N		= 20000;				% �w�K�p�f�[�^��
p		= n*n;					% �����̃A�g����(��ꐔ)
IterT	= 50;					% k-SVD�̊w�K��
qq		= 4;					% �����̍ۂɗp�����ꐔ

%	* ��ӂɌ��܂�p�����[�^ (�C�ӂ̃p�����[�^���玩���Ō��܂�)
d		= n^2;					% �~�j�o�b�`�̃s�N�Z����
DimTar	= n;					% Tramsform k-SVD �̊��T�C�Y

tic;


for j=1:length(ImNames)

	%   I           :   ���摜(256,256)
	%   In          :   I�Ƀm�C�Y������������
	%   X           :   I ���~�j�o�b�`���Ƃɏc�ɕ��ׂ�����
	%   Xn          :   In���~�j�o�b�`���Ƃɏc�ɕ��ׂ�����
	%   Xdn         :   Xn�̕��R�����𕽋ω���������
	%   Id0         :   Xn�̕��R������1�Ƃ�������
	%   IdxT        :   Xn�̃e�N�X�`��������1�Ƃ�������
	%   Xna         :   Xn�̃e�N�X�`�������𔲂��o�������̂��璼����������菜��������
	%   DCn         :   Xna�̒�������
	%   Xntrain     :   Xn�̃e�N�X�`�������𕽋ω���������
	%   Xntrain_r   :   Xn�������_���ɕ��ёւ�������

	% --- �摜�̓ǂݍ��� ---
% 	I	= double(imread(['./images/', ImNames{j},'.png']));	% �ǂݍ��݂�double�^�ϊ�
    t = Tiff(['./Standard Image Database/', ImNames{j},'.tif'],'r');
    I = read(t);
    I= I(:,:,1);
    I = imresize(I, [256, 256]);
    I = double(I);
	imshow(I,[0,255]);

	X	= im2col(I,[n,n],'sliding');					% �~�j�o�b�`(n�~n)���Ƃɉ摜���X���C�X�C��x�N�g���ŕ��ׂ�
	Xdn	= zeros(size(X));

	disp(['Image = ',ImNames{j}])

	for k=1:length(SigmaVec)

		% --- �G���d��摜�̐��� ---

		%	* �G���̐����Ɖ��Z
		sigma	= SigmaVec(k);							% SigmaVec �Ɏ��܂��Ă���G�����U���Ƃɉ摜����
        disp(['sigma = ', num2str(SigmaVec)]);
% 		In		= I + randn(size(I))*sigma;				% �摜�ɃK�E�X�����F�G�����d��
        In       = importdata(['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Noiseimg_��=', num2str(sigma), '_', ImNames{j}, '.txt']);
        
		%	* 2�����摜 -> �~�j�o�b�`�f�[�^
		Xn		= im2col(In,[n,n],'sliding');			% �~�j�o�b�`(n�~n)���ƂɎG���d��摜���X���C�X�C��x�N�g���ŕ��ׂ�

		%	* PSNR�̌v�Z
		PSNRin	= 10*log10(255.^2/mean((In(:)-I(:)).^2));	% ������PSNR
		figure,imshow(In,[0,255])
		title(['Noise Image : PSNR = ',num2str(PSNRin),' dB'])
		disp(['Initial PSNR [dB] : ',num2str(PSNRin)]);	% �\��
		

		% --- �����ݒ� ---
		Xdn		= zeros(size(Xn));						% ��s��
		Std0	= 1.15*sigma;							% ���R(<Std0)�����̌��o臒l
		Std1	= 1.5*sigma;							% �e�N�X�`��(>Std1)�����̌��o臒l
		cnt		= countcover([256 256],[n n],[1 1]);	% �I�[�o�[���b�v�񐔂̑���s��

        
        
		% --- �����w�K ---
        
		% �摜�𕽒R�����ƃe�N�X�`��(�\��)�����ɂ킯�āC�e�N�X�`�������݂̂�������w�K(?)

		%	* ���R�ƃe�N�X�`���𕪊�

		%	 - ���R�����̕����Ə���
		Id0			= find(std(Xn)<Std0);   			% ���U�� Std0 �ȉ��͕��R�����Ƃ݂Ȃ� ���R������1�Ƃ���
		Xdn(:,Id0)	= repmat(mean(Xn(:,Id0)),d,1);		% ���R�����𕽋ω������łȂ炷   %mean(Xn(:,Id0))��v�f�Ɏ���d�s1��̍s��𐶐�

		%	 - �e�N�X�`�������̕���
		IdxT		= find(std(Xn)>=Std1);				% ���U�� Std1 �ȏ�Ńe�N�X�`��(�\��)�Ƃ݂Ȃ� & ���̃C���f�b�N�X�ԍ��擾
		IdxN		= randperm(min(N, length(IdxT)));	% �e�N�X�`���̃C���f�b�N�X�ԍ��������_���œ���ւ� & N�̃e�N�X�`���𔲂��o��(N>�e�N�X�`����, �Ȃ炷�ׂẴe�N�X�`���𔲂��o��)
		Xntrain		= Xn(:,IdxN);						% �f�[�^����e�N�X�`���𔲂��o��


		%	* �e�N�X�`�����玫���w�K -> �����FOmega
% 		Omega		= TransformKSVD(Xntrain,DimTar,p,length(IdxN),IterT,qq);
        Omega       = importdata(['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Dic_noise_��=', num2str(sigma), '_', ImNames{j},'.txt']);

		%	* �����̐}��
		dic			= figure(1);
 		DisplayOmega(Omega, dic);						% �w�K����������\��
		title('Dictionary')
		disp(['Number of examples: ',num2str(size(Xntrain,2))]);


		% --- �摜���� ---
		% �摜�̊��S�ɕ��R�ȕ���(����)�Ƃ���ȊO�ɂ킯�āC���S�ɕ��R�ȕ����ȊO�𕜌�(?)

		%	* �摜�f�[�^���`
		Idx			= find(std(Xn)>=Std0);				% ���U�� Std0 �ȉ��͒����݂̂Ƃ݂Ȃ��ď������Ȃ� �C���f�b�N�X��ۑ�
		Xna			= Xn(:,Idx);						% �����݈̂ȊO�̃~�j�o�b�`�f�[�^�𒊏o

		%	* �~�j�o�b�`�f�[�^���璼������(���ϒl)������
		DCn			= mean(Xna);						% �~�j�o�b�`���Ƃ̕��ς��Z�o
		Xna			= Xna - repmat(DCn,d,1);			% ���ϒl�����������Ē�������������


% =========================================================================
%                            ADMM �ɂ��摜����
% =========================================================================
    
        T = zeros([1,1]);
        P = zeros([1,1]);
        S = zeros([1,1]);
        count = 1;
        
        for theta=0:0.1:1
       
            disp(['theta = ', num2str(theta)]);

            %	* ADMM�̌v�Z�ɕK�v�ȍs��̌v�Z

            Phy_coef	= eye(n*n);     % �W���̃X�p�[�X���Ɋւ���s��
            
%-----------�����v�Z�p�s��̍쐬--------------------------------------------

% % ----------������---------------------------------------------------------
% 
%             Phy_tv		= eye(n*n) - circshift(eye(n*n),[-1,0]);	% �����v�Z�p�s��
%             
%-----------������ + �c����-------------------------------------------------
%-----------������---------------------------------------------------------
            a1 = repmat(1,1,n*n);

            for i=1:1:n
                a1(:, n*i) = 0;
            end

            A1 = diag(-a1, 1);      %������
            B1  = diag(a1);         %�Ίp
            
            Phy_tv1 = (A1(1:n*n,1:n*n) + B1(1:n*n,1:n*n)) * Omega;     %   ������
            
            

%             disp(Phy_tv1);
%             dlmwrite('TV1.txt', Phy_tv1); 
%--------------------------------------------------------------------------       
%-----------�c����---------------------------------------------------------

            a2 = repmat(1,1,n*n);
            
            A2 = diag(-a2, n);      %�c����
            B2  = diag(a2);         %�Ίp

            A2(n*n-n+1:n*n,:) = 0;
            B2(n*n-n+1:n*n,:) = 0;
            
            Phy_tv2 = (A2(1:n*n,1:n*n) + B2(1:n*n,1:n*n)) * Omega;     %   �c����
            
%             disp(Phy_tv2);
%             dlmwrite('TV2.txt', Phy_tv2); 

%--------------------------------------------------------------------------

            Phy_tv  = [Phy_tv1; Phy_tv2];
            
% %             %   �O��
% %             a   = repmat(-0.25,1,n*n);
% %             
% %             %   ����
% %             b   = repmat(-0.25,1,n*n);
% %             
% %             for i=1:1:n*n
% %                 b(:, n*i) = 0;
% %             end
% %             
% %             %   �Ίp
% %             D   = eye(n*n);
% %             
% %             %   �Ίp�s��
% %             A1 = diag(a,n);
% %             A2 = diag(a,-n);
% %             B1 = diag(b,1);
% %             B2 = diag(b,-1);
% % 
% %             %   ��������TV����������
% %             Phy_tv  = A1(1:n*n,1:n*n) + A2(1:n*n,1:n*n) + B1(1:n*n,1:n*n) + B2(1:n*n,1:n*n) + D;	% �����v�Z�p�s��
% %             
% 
% %             disp(Phy_tv);
%             
% %--------------------------------------------------------------------------
            %	- �������s�� (���ӁI���Ԃ񐳂����Ȃ��D�����ŕς���)
            Dr = [Phy_coef; theta .* Phy_tv];

            
            %	* ADMM�ɂ�镜��
            lambda	= 6.7;

            w		= admm(Omega, Xna, Dr, lambda);		% ADMM�ɂ��X�p�[�X1�W�����擾
            Xdna	= Omega * w;						% �摜����

            %	* �~�j�o�b�`�f�[�^�ɒ�������(���ϒl)�����Z
            Xdna	= Xdna + repmat(DCn,d,1);				% �������������ɂ��ǂ�
            Xdn(:,Idx) = Xdna;

            %	* �~�j�o�b�`�f�[�^�����̂Q�����摜�f�[�^�ɕϊ�
            Idn		= col2imstep(Xdn,[256 256],[n n])./cnt;

            % --- �������� ---
            %	* PSNR�̌v�Z
            PSNRout = 10*log10(255.^2/mean((Idn(:)-I(:)).^2));
            disp(['Output PSNR [dB] : ',num2str(PSNRout)]);
            
            %	* SSIM�̌v�Z
            SSIM = ssim(Idn, I);
            disp(['Output SSIM : ',num2str(SSIM)]);

            %   * �����摜�̕\��
            figure,imshow(uint8(Idn),[0,255])	
            title(['Restroration Image : PSNR = ',num2str(PSNRout),' dB'])
            
            %   * �����摜�̕ۑ�
            rootpath = ['./result_image/proposal/theta/result/', ImNames{j},'/��=', num2str(sigma), '/'];   %�ۑ��t�@�C���p�X

%             restored_img = [rootpath,'theta=',num2str(theta), '.png']; % �t�@�C�����̍쐬
            restored_img = [rootpath,'SSIM_theta=',num2str(theta), '.png']; % SSIM

            saveas(gcf,restored_img) % �t�@�C���ւ̕ۑ�


            T(:,count) = theta;
            P(:,count) = PSNRout;
            S(:,count) = SSIM;
            
            count = count + 1;
            
        end
        
%         disp(T);
%         disp(P);
%         disp(S);

   
    
%-----------------------------PSNR�̌��ʏo��-------------------------------
%     %    * PSNR�O���t�̕\��,�ۑ�
%     path_graph_PSNR = [rootpath, 'PSNR_graph.png'];
%     plot(T,P);
%     saveas(gcf,path_graph_PSNR) % �t�@�C���ւ̕ۑ�
% 
%     %    * PSNR�̍ő�l�ƃɂ̒l
%     disp(['Image = ',ImNames{j}, '   sigma = ', num2str(sigma)]);
%     [bestPSNR, index_PSNR] = max(P);
%     disp(['bestPSNR = ', num2str(bestPSNR),'   theta  =', num2str(T(:,index_PSNR))]);
% 
%     %    * PSNR�̌��ʂ��t�@�C���ɏo��
%     A_PSNR = [T; P];
%     fileID = fopen([rootpath, 'result_PSNR.txt'],'w');
%     fprintf(fileID,'\n theta = %4.1f   bestPSNR = %7.5f\n',T(:,index_PSNR), bestPSNR);
%     fprintf(fileID,' �o�ߎ��� : %f\n\n', toc);
%     fprintf(fileID,'%9s �@%9s\n\n', 'theta', 'PSNR');
%     fprintf(fileID,'%9.1f �@%9.5f\n', A_PSNR);
%     fclose(fileID);
%     
    
    %------------------------------SSIM�̌��ʏo��------------------------------

%        * SSIM�O���t�̕\��,�ۑ�
    path_graph_SSIM = [rootpath, 'SSIM_graph.png'];
    plot(T,S);
    saveas(gcf,path_graph_SSIM) % �t�@�C���ւ̕ۑ�
    
%        * SSIM�̍ő�l�ƃɂ̒l
	disp(['Image = ',ImNames{j}])
    disp(['sigma = ', num2str(sigma)]);
    [bestSSIM, index_SSIM] = max(S);
    disp(['bestSSIM = ', num2str(bestSSIM),'   theta  =', num2str(T(:,index_SSIM))]);

%        * SSIM�̌��ʂ��t�@�C���ɏo��
    A_SSIM = [T; S];
    fileID = fopen([rootpath, 'result_SSIM.txt'],'w');
    fprintf(fileID,' theta = %4.1f   bestSSIM = %7.5f\n',T(:,index_SSIM), bestSSIM);
    fprintf(fileID,' �o�ߎ��� : %f\n\n', toc);
    fprintf(fileID,'%9s �@%9s\n\n', 'theta', 'SSIM');
    fprintf(fileID,'%9.1f �@%9.5f\n', A_SSIM);
    fclose(fileID);

    clear bestPSNR;
    clear bestSSIM;
    clear index;

    end
    
end

toc;

