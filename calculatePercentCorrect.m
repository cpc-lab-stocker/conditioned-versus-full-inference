%calculatePercentCorrect Calculate the percent correct.
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

function pcorrect = calculatePercentCorrect(th, thetaStim, pC, pthGC, ...
    PChGthetaLapse)
% assume PChGtheta_lapse corresponds to thetaStim
if length(pthGC)>length(thetaStim)
    pthGC = pthGC(:,ismember(th,thetaStim));
end
totalCateg = length(pC);
if size(pC,1) == 1
    pC = pC'; % rotate pC to be column vector
end
pcpthGc = repmat(pC,1,length(thetaStim)).*pthGC;
pcorrect = zeros(1,length(thetaStim));
for ic = 1:totalCateg
    pci(ic,:) = pcpthGc(ic,:)./sum(pcpthGc,1); % if sum 0, should just be 0
    pcorrect = pcorrect + PChGthetaLapse(ic,:).*pci(ic,:);
end