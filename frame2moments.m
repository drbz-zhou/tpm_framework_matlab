function moments = frame2moments(inF)
    [M, N] = size(inF);
    [x, y] = meshgrid(1:N, 1:M);
    x = x(:);
    y = y(:);
    F = inF(:);
    m.m00 = sum(F);
    if (m.m00 == 0)
        m.m00 = eps;
    end
    % The other central moments: 
    m.m10 = sum(x .* F);
    m.m01 = sum(y .* F);
    m.m11 = sum(x .* y .* F);
    m.m20 = sum(x.^2 .* F);
    m.m02 = sum(y.^2 .* F);
    m.m30 = sum(x.^3 .* F);
    m.m03 = sum(y.^3 .* F);
    m.m12 = sum(x .* y.^2 .* F);
    m.m21 = sum(x.^2 .* y .* F);
    moments = [double(m.m00), double(m.m10), double(m.m01), ...
        double(m.m11), double(m.m20), double(m.m02), double(m.m30),...
        double(m.m03), double(m.m12), double(m.m21)];
end