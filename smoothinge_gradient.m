function [grad] = smoothinge_gradient(X,y,W,b,alpha)

    sz = size(X);
    
    X1 = reshape(X,[sz(1)*sz(2),sz(3)]);
    W1 = reshape(W,[sz(1)*sz(2),1]);
    
    
    grad_tmp = zeros(sz(1)*sz(2),1);
    temp = X1'*W1 + b;
    z = y.*temp;

    z_ind1 = z<=0;
    y1 = y';
    grad_tmp = grad_tmp - sum(bsxfun(@times,y1(z_ind1),X1(:,z_ind1)),2);
    
    z_ind2 = 0<z & z<1;
    z_tmp = z(z_ind2).^(alpha);
    z_tmp1 = z_tmp';
    
    grad_tmp = grad_tmp + sum(bsxfun(@times,((z_tmp1-1).*y1(z_ind2)),X1(:,z_ind2)),2);
 
    grad = reshape(grad_tmp,[sz(1),sz(2)]);
end