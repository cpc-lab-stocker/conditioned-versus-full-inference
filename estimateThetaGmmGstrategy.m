%estimateThetaGmmGstrategy Estimate theta given sensory evidence mm.
%   [] = estimateThetaGmmGstrategy
%   Input:
%       strategy  Either 0 to use full inference strategy, or 1 to use
%                 conditioned inference.
%       noise     stdSensory and stdMemory.
%       thetaStim Stimulus theta known to the experimenter.
%       th        th list. 
%       pth       Prior over theta.
%       pC        Prior over category.
%       pthGC     p(th|category).
%
%   Output:
%       EthChc1   \hat{theta}(mm) when \hat{C} is C1.
%       EthChc2   \hat{theta}(mm) when \hat{C} is C2.
%
%   Examples see compareFullConditionInference.m

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

function [EthChc1, EthChc2, mm, pmmGm, pmGth, PChGm, PChGthetaLapse, ...
    m, PCGm] = estimateThetaGmmGstrategy(...
    strategy, stdSensory, stdMemory, thetaStim, th, pth, pC, pthGC)

% if full inference strategy, use pthGC(1/2,:) both the same as p(th)
if strategy == 0
    pthGC(1,:) = pth;
    pthGC(2,:) = pth;
end
nth = length(th);

% -------------------------------------------------------------------------
% layout the space based on the noise levels
% -------------------------------------------------------------------------
rangeM = [min(thetaStim)-5*stdSensory ...
    max(thetaStim)+5*stdSensory];
if rangeM(2) < max(th)
    rangeM = [min(th), max(th)];
end
nm = 1000;
m = linspace(rangeM(1), rangeM(2), nm);

nmm = 1000;
rangeMM = [min(rangeM)-6*stdMemory max(rangeM)+6*stdMemory];
if rangeMM(2) < max(th)
    rangeMM = [min(th), max(th)];
end
mm = linspace(rangeMM(1), rangeMM(2), nmm);

M = repmat(m',1,nth);
MM_m = repmat(mm',1,nm);
MM_th = repmat(mm',1,nth);
THm = repmat(th, nm, 1);  %th
THmm = repmat(th, nmm, 1);%th

% -------------------------------------------------------------------------
% Generative (forward)
% -------------------------------------------------------------------------
% p(m|th) given sensory noise
if stdSensory<1e-15
    pmGth = zeros(nm,nth);
    [~,m_ind] = min(abs(M-repmat(th, nm, 1)));
    for ith = 1:nth
        pmGth(m_ind(ith),ith) = 1;
    end
else
    pmGth = exp(-((M-THm).^2)./(2*stdSensory^2));
end
pmGth = pmGth./(repmat(sum(pmGth,1),nm,1));
% p(mm|m) given memory noise (nmmxnm)
if stdMemory<1e-15        %when memory noise close to 0, identity matrix
    pmmGm = zeros(nmm,nm);
    [~,mm_ind] = min(abs(MM_m-repmat(m, nmm, 1)));
    for imm = 1:nm
        pmmGm(mm_ind(imm),imm) = 1;
    end
else
    pmmGm = exp(-((MM_m-repmat(m, nmm, 1)).^2)./(2*stdMemory^2));
end
pmmGm = pmmGm./(repmat(sum(pmmGm,1),nmm,1));

% -------------------------------------------------------------------------
% Inference
% -------------------------------------------------------------------------
% 1. Categorical judgment
lapseRate = 0;
PCGm = (pthGC * pmGth') .* repmat(pC,1,nm);
% fix the issue when sensory noise is too low (from Luu's original notes)
indFirstNonZero = find(PCGm(1,:), 1);
PCGm(1, 1: indFirstNonZero-1) = PCGm(1, indFirstNonZero);
indLastNonZero = find(PCGm(2,:), 1, 'last');
PCGm(2, indLastNonZero+1:end) = PCGm(2, indLastNonZero);
PCGm = PCGm./(repmat(sum(PCGm,1),2,1));%normalize
% max posterior decision
PChGm = round(PCGm);
% more fix to substitude NaNs in PChGm (when stdSensory=0)
while ~isempty(find(isnan(PChGm)))
    %find ind with nan, but its previous one (ind-1) is not nan
    tmp = setdiff(find(isnan(PChGm(1,:)))-1,find(isnan(PChGm(1,:))));
    PChGm(:,tmp+1) = PChGm(:,tmp);
end
% marginalization to get p(\hat{c}|\theta)
PChGtheta = PChGm * pmGth(:, ismember(th, thetaStim));
PChGthetaLapse = lapseRate + (1 - 2*lapseRate) * PChGtheta;
PChGthetaLapse = PChGthetaLapse ./ repmat(sum(PChGthetaLapse, 1), 2, 1);

% 2. Estimate theta
pmmGth = exp(-((MM_th-THmm).^2)./(2*(stdSensory^2 + stdMemory^2)));
% p(mm|th) = N(th, sm^2 + smm^2)
pmmGth = pmmGth./(repmat(sum(pmmGth,1),nmm,1));
%posterior
pthGmmChc1 = pmmGth.*repmat(pthGC(1,:),nmm,1);
pthGmmChc2 = pmmGth.*repmat(pthGC(2,:),nmm,1);

% L2 loss (posterior mean)
%normalize
pthGmmChc1 = (pthGmmChc1./repmat(sum(pthGmmChc1,2),1,nth))';
%normalize and transpose to have mm along the horizontal axis
pthGmmChc2 = (pthGmmChc2./repmat(sum(pthGmmChc2,2),1,nth))'; %normalize
%theta_hat|cw,mm
EthChc1 = th * pthGmmChc1;
EthChc2 = th * pthGmmChc2;
