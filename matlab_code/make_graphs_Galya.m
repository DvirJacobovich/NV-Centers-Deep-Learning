% get l1 graphs
% load('end_of_sim')
load('..\COmpSENS\2\MC_homotopy_trial_100G_20theta_70phi.mat');
save_loc = 'C:\Users\Dvir\Desktop\NV-centers CompressedLearning\compressedLearning-main\COmpSENS\2';
file_prefix = 'fig_';
for i = 1:length(statistics)
    mask = mod(statistics(i).meas,5)==0;
   %%%%%%%%%%%%%%%%%%%%%%%%commented out for CS_analysis_inner_calc
%     if statistics(i).noise ==0
%         noise_str = '0';
%     else
%         noise_str = num2str(statistics(i).noise);
%         noise_str = noise_str(3:end);
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     mask = zeros(size(CS.meas_num));
%    std_errs = [];
%    k=0;
%    for m=[26,30:5:num_recons]
%        k=k+1;
%        mask = or(mask,(meas_num==m));
%        std_errs(k) = std(mean_pk_err_locations_with_unc(meas_num==m));
%    end
%    figure
%    subplot(1,2,1) 
%    
%    boxplot(mean_pk_err_locations_with_unc(mask),meas_num(mask),'positions',sort(unique(meas_num(mask))),...
%        'PlotStyle','compact','symbol','')
%    
%    grid on
%    title([folder_str ', Mean Error in Peak Location: ' num2str(curr_freq) ...
%        ' Frequencies'])
%    xlabel('# of Measurements')
%    ylabel('Frequency (MHz)')
%    %ylim([0,3])
%    subplot(1,2,2) 
%    
%    a=histogram(meas_num,(26-0.5):(100+0.5));
    
%    title([ ' Convergence Probability'])
%    xlabel('# of Measurements')
%    ylabel('%')
%    hist_data(:,1)=round(a.BinEdges(1:end-1));
%    hist_data(:,2)=a.Values;
%    Normalized_err=mean_pk_err_locations_with_unc;
if i == 1
    min_1 = 0;
    max_1 = 0;
else
    min_1 = min(statistics.meas);
    max_1 = max(statistics.meas);
end

   for s=min_1:max_1
       b=(find(statistics.meas==s));
%        Normalized_err(b)=mean_pk_err_locations_with_unc(b)./(hist_data((s-25),2)/100);
       err_per_freq_num(s-(min_1-1))=mean(statistics(1).mean_err(b));
       statistics_on_freq_num(s-(min_1-1))=length(b);
       freq_num_per(s-(min_1-1)) = s;
   end
   figure;plot(freq_num_per,err_per_freq_num,'.')

   title(['Mean error in MHz, \sigma=' num2str(statistics(i).noise) ', ' ...
       num2str(statistics(i).freq) ' frequencies' ])
   xlabel('# of Measurements')
   ylabel('Error(MHz)')
   name_err=['CS_simulation_Err_MHz_120g_0.009Noise.png'];
   name_err2=['CS_simulation_Err_MHz_120g_0.009Noise.fig'];
   saveas(gcf,name_err)
   savefig(gcf,name_err2,'compact')
    
   
   figure;plot(freq_num_per,statistics_on_freq_num,'.')
   title(['Success probability, \sigma=' num2str(statistics(i).noise) ', ' ...
        num2str(statistics(i).freq) ' frequencies' ]);
   xlabel('# of Measurements')
   ylabel('Success')
   name_err=['CS_simulation_Success probability_120g_0.009Noise.png'];
   name_err2=['CS_simulation_Success probability_120g_0.009Noise.fig'];
   saveas(gcf,name_err)
   savefig(gcf,name_err2,'compact')
   
   meas_time=500e-6*200; %time of 1 measurement in s
   err_in_Gauss=err_per_freq_num/2.87;
   sensitivity=err_in_Gauss.*sqrt(2*meas_time.*freq_num_per);
   Norm_sensitivity=err_in_Gauss.*sqrt(2*meas_time.*(freq_num_per./(statistics_on_freq_num/100))); %the 2 is for ref
   
   figure;plot(freq_num_per,err_in_Gauss,'.')
   title(['Mean error in Gauss ,\sigma=' num2str(statistics(i).noise) ', ' ...
        num2str(statistics(i).freq) ' frequencies' ]);
   xlabel('# of Measurements','FontSize',12)
   ylabel('Error [G]','FontSize',12)
   name_err=['CS_simulation_Err_Gauss_120g_0.009Noise.png'];
   name_err2=['CS_simulation_Err_Gauss_120g_0.009Noise.fig'];
   saveas(gcf,name_err)
   savefig(gcf,name_err2,'compact')
   
   figure;plot(freq_num_per,sensitivity,'.')
   title('Sensitivity [G*sqrt(Hz)] ','FontSize',12);
   xlabel('# of Measurements','FontSize',12)
   ylabel('G/sqrt(Hz)','FontSize',12)
   name_err=['CS_simulation_Sensitivity_120g_0.009Noise.png'];
   name_err2=['CS_simulation_Sensitivity_120g_0.009Noise.fig'];
   saveas(gcf,name_err)
   savefig(gcf,name_err2,'compact')
   
   
   figure;plot(freq_num_per,Norm_sensitivity,'.')
   title(['Normalized Sensitivity, \sigma=' num2str(statistics(i).noise) ', ' ...
        num2str(statistics(i).freq) ' frequencies' ]);
   xlabel('# of Measurements','FontSize',12)
   ylabel('[G\sqrt(Hz)] , time normalized in probability','FontSize',12)
   name_err=['CS_simulation_Normalized_Sensitivity_120g_0.009Noise.png'];
   name_err2=['CS_simulation_Normalized_Sensitivity_120g_0.009Noise.fig'];
   saveas(gcf,name_err)
   savefig(gcf,name_err2,'compact')
   
    
    close all
end