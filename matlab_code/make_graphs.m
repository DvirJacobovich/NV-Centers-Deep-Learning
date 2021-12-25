% get l1 graphs
%load('end_of_sim')
load('..\COmpSENS\2MC_homotopy_trial_100G_20theta_70phi.mat')
% save_loc = 'C:\Users\Owner\Desktop\COmpSENS\2/';
save_loc = 'C:\Users\Dvir\Desktop\NV-centers CompressedLearning\compressedLearning-main\COmpSENS\2';

file_prefix = 'fig_';
for i = 1:length(statistics)
    mask = mod(statistics(i).meas,5)==0;
    figure
    subplot(1,2,1)
    boxplot(statistics(i).mean_err(mask),statistics(i).meas(mask),...
        'PlotStyle','compact','symbol','')
    grid on
    
%     if i<12
%         ylim([0 5])
%     else
%         ylim([0 10])
%     end
    ylim([0 5])
    xlabel('Measurement Number')
    ylabel('Mean Error (MHz)')
    title(['\sigma=' num2str(statistics(i).noise) ', ' ...
        num2str(statistics(i).freq) ' frequencies' ])
    %axis tight
    subplot(1,2,2)
    plot(1:max_projs,statistics(i).prob/num_trials)
    grid on
    xlabel('Measurement Number')
    ylabel('Probability')
    title('Convergence Probability')
    ylim([0 1])
    xlim([1 max_projs])
    %axis tight    
    set(gcf,'Position',[119 246 1050 420])
    %pause(3)
    if statistics(i).noise ==0
        noise_str = '0';
    else
        noise_str = num2str(statistics(i).noise);
        noise_str = noise_str(3:end);
    end
    file_title = ['statistics_' num2str(data.B_mag) 'G_' num2str(data.B_theta) 'theta_' num2str(data.B_phi) 'phi_' ...
        'sigma_0pt' noise_str '_' num2str(statistics(i).freq) 'freq'];
    full_loc = [save_loc file_prefix file_title];
    savefig(gcf,full_loc,'compact')
    saveas(gcf,[full_loc '.png'])
    close all
end