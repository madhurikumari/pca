%mypca.m
%Use Principal Component Analysis Method to classify digit from MNIST
%Database

clear;
load mnistdata;


basis_len = 5; %choose basis length
Us=zeros( 28*28, basis_len, 10);
success = zeros(10,1);
for k=1:10
    %create training set T
    s = strcat('train',num2str(k-1));
    A = double(eval(s));
    [m,~]=size(A);
	[U,~,~] = svds( A', basis_len );% get first 5 singular vector
    Us(:,:,k)=U;%basis space
end

matches = sum(O(:,1)== num2str(k-1))
success(k,1) = matches/m;
disp(success)

function O=mkpca(A,Us)
%Inputs:
%n-by-784 matrix A containing n digits
%784-by-k-by-10 matrix T containing the first 
%   k singular vectors for each of the training sets 
%   (train0',train1',train2'...,train9') (tranposed!)
%Outputs:
%   O = n x 1 classified numbers

        [m,~]=size(A);
        O = zeros(m,1);
        
        %distance
        for i=1:m %for each row of A
            z = double(A(i,:))';
            dist = zeros(10,1);
            for k=1:10
                %find 2-norm distance to each digit
                Uk = Us(:,:,k);
                dist(k) = norm( z - Uk*(Uk'*z) )
            end
            
            [~,I] = min(dist);%min dist is closest digit match
            O(i,1)=I-1; % minus 1 because digits are 0-9
        end

       
    end

