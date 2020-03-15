
%	Transform k-SVD ��p���������w�K��OMP(orADMM)��p�����摜����

clear

close all

% --- �p�����[�^�Q ---

%	* �摜�̐ݒ�
% ImNames={'house'};				% (�摜��).png�C�����w�肷��Ɖ摜���ƂɎG���������s���D
ImNames={'cameraman', 'house', 'jetplane', 'lake', 'lena', 'livingroom','mandril','peppers', 'pirate', 'walkbridge', 'woman_blonde', 'woman_darkhair'};	% �摜�𕡐��w�肵����

% SigmaVec=[15];					% �G�����U�C�����w�肷��Ƃ��ꂲ�ƂɎG���������s���D
% SigmaVec=[5, 10, 15, 20, 25, 30];		% �G�����U�𕡐��w�肵����
SigmaVec=[35, 40];		% �G�����U�𕡐��w�肵����

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
    imshow(uint8(I));
    
	X	= im2col(I,[n,n],'sliding');					% �~�j�o�b�`(n�~n)���Ƃɉ摜���X���C�X�C��x�N�g���ŕ��ׂ�
	Xdn	= zeros(size(X));

	disp(['Image = ',ImNames{j}]);

	for k=1:length(SigmaVec)

		% --- �G���d��摜�̐��� ---

		%	* �G���̐����Ɖ��Z
		sigma	= SigmaVec(k);							% SigmaVec �Ɏ��܂��Ă���G�����U���Ƃɉ摜����
        disp(['sigma = ', num2str(sigma)]);
        In		= I + randn(size(I))*sigma;				% �摜�ɃK�E�X�����F�G�����d��
        dlmwrite(['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Noiseimg_��=', num2str(sigma), '_', ImNames{j}, '.txt'], In);                   %�G���摜��ۑ�(txt�`��)

		%	* 2�����摜 -> �~�j�o�b�`�f�[�^
		Xn		= im2col(In,[n,n],'sliding');			% �~�j�o�b�`(n�~n)���ƂɎG���d��摜���X���C�X�C��x�N�g���ŕ��ׂ�       

		%	* PSNR�̌v�Z
		PSNRin	= 10*log10(255.^2/mean((In(:)-I(:)).^2));	% ������PSNR
        
		figure,imshow(In,[0,255])
		title(['Noise Image : PSNR = ',num2str(PSNRin),' dB'])
		disp(['Initial PSNR [dB] : ',num2str(PSNRin)]);	% �\��
        saveas(gcf, ['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Noiseimg_��=', num2str(sigma), '_', ImNames{j}, '.png']);                     %�G���摜��ۑ�(png�`��)

		% --- �����ݒ� ---
		Xdn		= zeros(size(Xn));						% ��s��
		Std0	= 1.15*sigma;							% ���R(<Std0)�����̌��o臒l
		Std1	= 1.5*sigma;							% �e�N�X�`��(>Std1)�����̌��o臒l
		cnt		= countcover([256 256],[n n],[1 1]);	% �I�[�o�[���b�v�񐔂̑���s��
        
        
		% --- �����w�K ---
        
		% �摜�𕽒R�����ƃe�N�X�`��(�\��)�����ɂ킯�āC�e�N�X�`�������݂̂�������w�K(?)

		%	* ���R�ƃe�N�X�`���𕪊�

        %   �G���摜���玫���𐶐�
		%	 - ���R�����̕����Ə���
		Id0			= find(std(Xn)<Std0);   			% ���U�� Std0 �ȉ��͕��R�����Ƃ݂Ȃ� ���R������1�Ƃ���
		Xdn(:,Id0)	= repmat(mean(Xn(:,Id0)),d,1);		% ���R�����𕽋ω������łȂ炷   %mean(Xn(:,Id0))��v�f�Ɏ���d�s1��̍s��𐶐�

		%	 - �e�N�X�`�������̕���
		IdxT		= find(std(Xn)>=Std1);				% ���U�� Std1 �ȏ�Ńe�N�X�`��(�\��)�Ƃ݂Ȃ� & ���̃C���f�b�N�X�ԍ��擾
		IdxN		= randperm(min(N, length(IdxT)));	% �e�N�X�`���̃C���f�b�N�X�ԍ��������_���œ���ւ� & N�̃e�N�X�`���𔲂��o��(N>�e�N�X�`����, �Ȃ炷�ׂẴe�N�X�`���𔲂��o��)
		Xntrain		= Xn(:,IdxN);						% �f�[�^����e�N�X�`���𔲂��o��

%         %   ���摜���玫���𐶐�
%         %	 - ���R�����̕����Ə���
%         Id0			= find(std(X)<Std0);   			% ���U�� Std0 �ȉ��͕��R�����Ƃ݂Ȃ� ���R������1�Ƃ���
% 		Xdn(:,Id0)	= repmat(mean(X(:,Id0)),d,1);		% ���R�����𕽋ω������łȂ炷   %mean(Xn(:,Id0))��v�f�Ɏ���d�s1��̍s��𐶐�
% 
%         %	 - �e�N�X�`�������̕���
% 		IdxT		= find(std(X)>=Std1);				% ���U�� Std1 �ȏ�Ńe�N�X�`��(�\��)�Ƃ݂Ȃ� & ���̃C���f�b�N�X�ԍ��擾
% 		IdxN		= randperm(min(N, length(IdxT)));	% �e�N�X�`���̃C���f�b�N�X�ԍ��������_���œ���ւ� & N�̃e�N�X�`���𔲂��o��(N>�e�N�X�`����, �Ȃ炷�ׂẴe�N�X�`���𔲂��o��)
% 		Xntrain		= X(:,IdxN);						% �f�[�^����e�N�X�`���𔲂��o��


		%	* �e�N�X�`�����玫���w�K -> �����FOmega
		Omega		= TransformKSVD(Xntrain,DimTar,p,length(IdxN),IterT,qq);
        dlmwrite(['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Dic_noise_��=', num2str(sigma), '_', ImNames{j},'.txt'], Omega);                     %�G���摜�������������̕ۑ�

        
		%	* �����̐}��
		dic			= figure(1);
 		DisplayOmega(Omega, dic);						% �w�K����������\��
		title('Dictionary')
		disp(['Number of examples: ',num2str(size(Xntrain,2))]);
        saveas(gcf, ['./dataset/', ImNames{j}, '/��=', num2str(sigma), '/Dic_noise_��=', num2str(sigma), '_', ImNames{j},'.png']);                     %�G���摜��ۑ�(png�`��)
       
    end
    
end


toc;

