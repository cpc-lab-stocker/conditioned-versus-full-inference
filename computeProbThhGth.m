%computeProbThhGth Change of variable to get the estimate distribution.
% p(\hat{theta}|theta) is calcuated based on p(mm|theta) and
% \hat{theta}(mm).
%
% Examples see compareFullConditionInference.m

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

function [pthhGth] = computeProbThhGth(th, thetaStim, ...
    EthChcw, EthChccw, mm, pmmGm, pmGth, PChGm, PChGthetaLapse)

nmm = length(mm); 
nth = length(th); 
% discard repeating/decreasing values (required for interpolation)
indKeepCw_init = find(~isnan(EthChcw));
indKeepCw = 1:length(EthChcw);
while sum(diff(EthChcw)<=0) >0
    indDiscardCw = [false diff(EthChcw)<=0];
    EthChcw(indDiscardCw) = [];
    indKeepCw(indDiscardCw) = [];
end
indKeepCw = sort(intersect(indKeepCw,indKeepCw_init));
EthChcw(find(isnan(EthChcw))) = [];

indKeepCcw_init = find(~isnan(EthChccw));
indKeepCcw = 1:length(EthChccw);
while sum(diff(EthChccw)<=0) >0
    indDiscardCcw = [diff(EthChccw)<=0 false];
    EthChccw(indDiscardCcw) = [];
    indKeepCcw(indDiscardCcw) = [];
end
indKeepCcw = sort(intersect(indKeepCcw,indKeepCcw_init));
EthChccw(find(isnan(EthChccw))) = [];

dstep = .1;
a = 1./gradient(EthChcw,dstep);
% attention marginalization:
% compute distribution only over those ms that lead to cw decision!
pmmGthChcw = pmmGm * (pmGth(:, ismember(th, thetaStim)).*...
    repmat(PChGm(1,:)',1,length(thetaStim)));
pmmGthChcw = pmmGthChcw ./ repmat(sum(pmmGthChcw,1),nmm,1);
b = repmat(a',1,length(thetaStim)) .* pmmGthChcw(indKeepCw, :);
pthhGthChcw = interp1(EthChcw,b,th,'linear','extrap');
pthhGthChcw(pthhGthChcw < 0) = 0;

a = 1./gradient(EthChccw,dstep);
pmmGthChccw = pmmGm * (pmGth(:, ismember(th, thetaStim)).*...
    repmat(PChGm(2,:)',1,length(thetaStim)));
pmmGthChccw = pmmGthChccw ./ repmat(sum(pmmGthChccw,1),nmm,1);
b = repmat(a',1,length(thetaStim)) .* pmmGthChccw(indKeepCcw, :);
pthhGthChccw = interp1(EthChccw,b,th,'linear','extrap');
pthhGthChccw(pthhGthChccw < 0) = 0;

pthhGthChcw = pthhGthChcw./repmat(sum(pthhGthChcw,1),nth,1);
% normalize - especially needed if conv2 for motor noise;
%             after interp1 supposed to be normalized to 1.
pthhGthChccw = pthhGthChccw./repmat(sum(pthhGthChccw,1),nth,1);

pthhGthChcw(isnan(pthhGthChcw)) = 0;
pthhGthChccw(isnan(pthhGthChccw)) = 0;

pthhGth = pthhGthChcw.*repmat(PChGthetaLapse(1,:),nth,1) + ...
    pthhGthChccw.*repmat(PChGthetaLapse(2,:),nth,1);

end

