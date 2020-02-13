%compareFullConditionInference Main function to compare inference strategies.
% compareFullConditionInference plots summary results of Gaussian priors
% that are equally likely.
% compareFullConditionInference(priorType) plots summary results of the
% priors type specified in priorType with following options: 'gaussian',
% 'box' (uniform box prior), and 'diffSDs' (categorical priors with the 
% same mean but different variance). For each type, both categories are 
% equally likely.   
% compareFullConditionInference(priorType, pC) pC is used to specify the
% prior ratio over the categories. For example, pC = [0.4; 0.6].
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

function compareFullConditionInference(priorType, pC)

if nargin<1
    priorType = 'gaussian'; 
    pC = [0.5; 0.5];
end
if nargin<2
    pC = [0.5; 0.5];
end

if size(pC,2)>1
    pC = pC'; % to be column vector
end
pC = pC/sum(pC);

% stimulus orientations
thetaStim = -15:.1:15;
dstep = 0.1;
thetaStim = round(thetaStim, -log10(dstep));

% categorical priors
[th, nth, pthGC, pth] = generateCategoricalPriors(...
    priorType, pC, dstep);

% th and thetaStim matrix for calculating MSE
mseMtrxStim = repmat(thetaStim,nth,1);
mseMtrxTh = repmat(th',1,length(thetaStim));
mseMtrx = (mseMtrxStim-mseMtrxTh).^2;

% levels of noise to be tested
stdSensoryAll = 0:.2:4;
stdMemoryAll = 0:.2:4;
overallRatio = nan(length(stdSensoryAll),length(stdMemoryAll));

iflogratio = 1; % shown in log ratio
noChange = double(~iflogratio);
isensory = 0;
for stdSensory = stdSensoryAll
    isensory = isensory + 1;
    imemory = 0;
    for stdMemory = stdMemoryAll
        imemory = imemory + 1;
        if ~(stdSensory || stdMemory)
            overallRatio(isensory,imemory) = noChange;
        else
            tic
            for strategy = [0, 1] % 0 - full inference; 1 - conditioned
                % ---------------------------------------------------------
                % estimate theta given mm under each strategy
                % ---------------------------------------------------------
                [EthChcw, EthChccw, mm, pmmGm, pmGth, PChGm, ...
                    PChGthetaLapse] = estimateThetaGmmGstrategy(strategy,...
                    stdSensory, stdMemory, thetaStim, th, pth, pC, pthGC);
                
                % ---------------------------------------------------------
                % change of variable to get estimate distribution
                % p(\hat{\theta}|theta)
                % ---------------------------------------------------------
                [pthhGth] = computeProbThhGth(...
                    th, thetaStim, EthChcw, EthChccw, mm, pmmGm, pmGth, ...
                    PChGm, PChGthetaLapse);
                
                % MSE(theta): only the corresponding row and column
                for istim = 1:length(thetaStim)
                    mse(strategy+1,istim) = ...
                        pthhGth(:,istim)'*mseMtrx(:,istim);
                end
            end
            
            pthetaStim = pth(ismember(th, thetaStim));
            pthetaStim = pthetaStim/sum(pthetaStim);
            mseAve = mse*pthetaStim';
            mseRatio = mseAve(2)./mseAve(1);
            if iflogratio
                mseRatio = log(mseRatio);
            end
            overallRatio(isensory,imemory) = mseRatio;
            
            toc % calculate MSE for each noise level
        end
    end
end
%% ------------------------------------------------------------------------
%  Visualize
%  ------------------------------------------------------------------------
fontsize = 20;
f = figure('position',[10 10 1000 300]); f.Renderer='Painters';
subplot(2,3,1); % prior
plot(th,pthGC.*repmat(pC,1,size(pthGC,2)))
xlim([min(thetaStim) max(thetaStim)]);
l=legend('$C_1$','$C_2$','Interpreter','latex');
legend('boxoff');
l.FontSize = fontsize-4; 
box off;
xlabel('$\theta$','Interpreter','latex','fontsize',fontsize);
ylabel('$p(C)p(\theta \mid C)$','Interpreter','latex','fontsize',fontsize);

% -------------------------------------------------------------------------
% examples of local performance ratio
% -------------------------------------------------------------------------
color_list = [127 229 229; 0 102 102]/255;
stdSensory = 2;
for stdMemory = [0, 3]
    for strategy = [0, 1]
        [EthChcw, EthChccw, mm, pmmGm, pmGth, PChGm, PChGthetaLapse] = ...
            estimateThetaGmmGstrategy(strategy, ...
            stdSensory, stdMemory, thetaStim, th, pth, pC, pthGC);
        [pthhGth] = computeProbThhGth(...
            th, thetaStim, EthChcw, EthChccw, mm, pmmGm, pmGth, ...
            PChGm, PChGthetaLapse);
        for istim = 1:length(thetaStim)
            mse(strategy+1,istim) = pthhGth(:,istim)'*mseMtrx(:,istim);
        end
    end
    if ~stdMemory % curve of percent correct (at sensory stage)
        subplot(2,3,4); hold on;
        pCorrect = calculatePercentCorrect(th, thetaStim, pC, pthGC, ...
            PChGthetaLapse);
        plot(thetaStim,pCorrect,'k-','linewidth',2)
        ylim([0,1]); yticks(0:.2:1);
        ylabel('$p(\hat{C}$ correct)','Interpreter','latex',...
            'fontsize',fontsize);
        xlim([min(thetaStim) max(thetaStim)]);
        xlabel('$\theta$','Interpreter','latex','fontsize',fontsize);
    end
    mseRatio = mse(2,:)./mse(1,:);
    
    subplot(1,3,2); hold on;
    plot(thetaStim,log(mseRatio),...
        'color',color_list(mod(stdMemory,2)+1,:),'linewidth',2)
    
end
subplot(1,3,2);
l=legend('No late noise','Late noise');
legend('autoupdate','off');
l.Location = 'northeast'; l.FontSize = fontsize-4; legend('boxoff');
xlim([min(thetaStim) max(thetaStim)]);
ylim(log([.4 2.5]))
yticks(log(.4:.3:2.5)); yticklabels(.4:.3:2.5)

plot([-15 15],[noChange, noChange],'k--')
xlabel('$\theta$','Interpreter','latex','fontsize',fontsize);
ylabel('MSE ratio','Interpreter','latex','fontsize',fontsize)

% -------------------------------------------------------------------------
% summary plot across multiple noise levels
% -------------------------------------------------------------------------
subplot(1,3,3); hold on;
imagescRange = log([.4 2.5]-noChange);
imagesc('xdata',stdMemoryAll,'ydata',stdSensoryAll,...
    'cdata',overallRatio,imagescRange)
colorbar('Ticks',log(.4:.3:2.5),'TickLabels',.4:.3:2.5,...
    'location','northoutside')
colormap(bluewhitered(256));
axis equal;
xlim([0 max(stdMemoryAll)]); xticks(0:4);
ylim([0 max(stdSensoryAll)]); yticks(0:4);
ylabel('$\sigma_s$','Interpreter','latex','fontsize',fontsize);
xlabel('$\sigma_m$','Interpreter','latex','fontsize',fontsize);

hold on; % label the local examples
plot(0,2,'o','markersize',12,'markeredgecolor','k',...
    'markerfacecolor',[127 229 229]/255,'linewidth',2)
plot(3,2,'o','markersize',12,'markeredgecolor','k',...
    'markerfacecolor',[0 102 102]/255,'linewidth',2)

