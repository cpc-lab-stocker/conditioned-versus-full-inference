%generateCategoricalPriors Generate categorical priors for two categories. 
% Available priorType includes: 'gaussian', 'box', and 'diffSDs'.
%

% Created by Cheng Qiu.
% Copyright (c) 2020 cpc-lab-stocker
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [th, nth, pthGC, pth] = generateCategoricalPriors(...
    priorType, pC, dstep)
% default pC[.5, .5] and dstep[.1]
switch nargin
    case 1
        pC = [.5, .5]'; dstep = .1; 
    case 2
        dstep = .1; 
end

% set the range and grid for th
rangeth = [-42 42]; % theta range 
th = rangeth(1):dstep:rangeth(2);
th = round(th, -log10(dstep));
nth = length(th);

% categorical prior parameters based on priorType
switch priorType
    case 'box'      % non-overlapping flat prior
        c_ref = 0; pthcw = 12; pthccw = -12;
    case 'gaussian' % overlapping gaussian prior
        c1_0 = -3; c2_0 = 3; sigma_c1 = 3; sigma_c2 = 3;
    case 'diffSDs'  % priors as in Qamer 2013
        c1_0 = 0; c2_0 = 0; sigma_c1 = 1.5; sigma_c2 = 6;
end

% calculate prior once th grid is set
switch priorType
    case 'box'
        % non-overlapping categorical prior
        pthGC = zeros(2,nth);
        pthGC(1,:) = zeros(size(th));
        pthGC(1,intersect(find(th<=c_ref),find(th>pthccw))) = 1;
        pthGC(2,:) = zeros(size(th));
        pthGC(2,intersect(find(th>=c_ref),find(th<pthcw))) = 1;
        % p(th) integrated over C
        pth = pC(1)*pthGC(1,:)+pC(2)*pthGC(2,:);
        pth(th==c_ref) = 0; pth(th==c_ref) = max(pth);
    otherwise
        pthGC(1,:) = normpdf(th,c1_0,sigma_c1);
        pthGC(2,:) = normpdf(th,c2_0,sigma_c2);
        pth = pC(1)*pthGC(1,:)+pC(2)*pthGC(2,:);
end
end

