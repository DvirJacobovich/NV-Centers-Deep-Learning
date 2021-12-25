%% addpaths
addpath 'C:\Users\Dvir\Desktop\NV-centers Compressed learning\NV-centers Compressed learning\compressedLearning-main\L1_homotopy_v2.0'
%% Path to change and sim parameters to try
location = 'C:\Users\Dvir\Desktop\NV-centers Compressed learning\NV-centers Compressed learning\compressedLearning-main\L1_homotopy_v2.0';
n = 500;
m = 1500;
COUNTER = 0;

% freqs_to_try = ones(1, n) * 3;%300 samples of 3 freqs together. Usages: 3 or [2,3,4];
freqs_to_try = 4;
base_noise = 0.0004;%[0:0.0003:0.0009, 0.001:0.002:0.008];   
% base_noise = [0:0.0003:0.0009, 0.001:0.002:0.008];  
num_trials=1;

%% Get diamond simulation
for j = n : m
contrast = 0.002;
data = mock_diamond2();
% figure;plot(data.smp_freqs,data.target)  %Plots target data
target_num_pks = length(data.peak_locs); 
freqs = data.smp_freqs;
%file_title_sim = [location 'diamond' num2str(data.B_mag) 'G_' num2str(data.B_theta) 'theta_' num2str(data.B_phi) 'phi'];
%save(file_title_sim,'data');

window_start = min(data.peak_locs)-10;
window_end = max(data.peak_locs)+10;


%% initialize strucutre for saving data
CS = spike_adaptive_l1homotopy(freqs, 1, ...
    1, window_start, window_end, data);
min_projs = CS.num_initial_projs;
max_projs = CS.num_max_projs;
meas_of_interest = min_projs:max_projs;

statistics = struct();
statistics(length(base_noise)*length(freqs_to_try)).mean_err = [];
statistics(length(base_noise)*length(freqs_to_try)).meas = [];
for i=1:length(base_noise)*length(freqs_to_try)
    statistics(i).prob = zeros(CS.num_max_projs,1);
end

%% Main Monte Carlo Loop
q=0;
t = cputime;
for curr_noise = base_noise
    data.sigma = curr_noise;
    for num_simulataneous_freqs = freqs_to_try
        q=q+1;
        k=0;
        m=0;
        for i=1:num_trials
            CS = spike_adaptive_l1homotopy(data.smp_freqs, num_simulataneous_freqs, ...
                contrast, window_start, window_end, data);
%             CS.display = false;
            while CS.Continue
                [CS,current_freq] = CS.frequency();
                data = data.getMeasurement(current_freq);
                CS = CS.process_data(data.signal);
            end
            if ~isempty(CS.meas_num)
                statistics(q).prob(CS.meas_num) = statistics(q).prob(CS.meas_num)+1;
                statistics(q).mean_err = [statistics(q).mean_err; CS.actual_err];
                statistics(q).meas = [statistics(q).meas; CS.meas_num];
                statistics(q).noise = curr_noise;
                statistics(q).freq = num_simulataneous_freqs;
            end           
            close all
            if mod(i,10)==0
                disp(['Done with trial ' num2str(i) ' out of ' num2str(num_trials) ' trials.'])
            end
        end
        disp(['Done with ' num2str(num_simulataneous_freqs) ' frequencies'])
        
        
% make_graphs

    end
    disp(['Done with noise=' num2str(curr_noise)])
    % start here
%     mask = zeros(size(CS.meas_num));
%    std_errs = [];
%    k=0;
%    num_recons = 
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
% %    title([folder_str ', Mean Error in Peak Location: ' num2str(curr_freq) ...
% %        ' Frequencies'])
%    xlabel('# of Measurements')
%    ylabel('Frequency (MHz)')
%    %ylim([0,3])
%    subplot(1,2,2) 
%    
%    a=histogram(meas_num,(26-0.5):(100+0.5));
%     
%    title([ ' Convergence Probability'])
%    xlabel('# of Measurements')
%    ylabel('%')
%    hist_data(:,1)=round(a.BinEdges(1:end-1));
%    hist_data(:,2)=a.Values;
% %    Normalized_err=mean_pk_err_locations_with_unc;
%    for s=26:120
%        b=(find(meas_num==s));
% %        Normalized_err(b)=mean_pk_err_locations_with_unc(b)./(hist_data((s-25),2)/100);
%        err_per_freq_num(s-25)=mean(mean_pk_err_locations_with_unc(b));
%        statistics_on_freq_num(s-25)=length(b);
%        freq_num_per(s-25) = s;
%    end
%    figure;plot(freq_num_per,err_per_freq_num,'.')
%    title('Mean error in MHz ');
%    xlabel('# of Measurements')
%    ylabel('Error(MHz)')
%    figure;plot(freq_num_per,statistics_on_freq_num,'.')
%    title('Success probability');
%    xlabel('# of Measurements')
%    ylabel('Success')
%    
%    meas_time=10e-3; %time of 1 measurement in s
%    err_in_Gauss=err_per_freq_num/2.87;
%    sensitivity=err_in_Gauss.*sqrt(2*meas_time.*freq_num_per);
%    Norm_sensitivity=err_in_Gauss.*sqrt(2*meas_time.*(freq_num_per./(statistics_on_freq_num/100))); %the 2 is for ref
%    figure;plot(freq_num_per,err_in_Gauss,'.')
%    title('Mean error in Gauss ','FontSize',12);
%    xlabel('# of Measurements','FontSize',12)
%    ylabel('Error [G]','FontSize',12)
%    figure;plot(freq_num_per,sensitivity,'.')
%    title('Sensitivity [G/sqrt(Hz)] ','FontSize',12);
%    xlabel('# of Measurements','FontSize',12)
%    ylabel('G\sqrt(Hz)','FontSize',12)
%    figure;plot(freq_num_per,Norm_sensitivity,'.')
%    title('Normalized Sensitivity','FontSize',12);
%    xlabel('# of Measurements','FontSize',12)
%    ylabel('[G\sqrt(Hz)] , time normalized in probability','FontSize',12)
   
end
    %% save data
%         var_title = [location 'MC_homotopy_trial_'  num2str(data.B_mag) 'G_' num2str(data.B_theta)...
%             'theta_' num2str(data.B_phi) 'phi.mat'];

%         location = 'C:\Users\owner\Desktop\Dvir\NV-centers Compressed learning\NV-centers CompressedLearning-20211121T154621Z-001\compressedLearning-main\COmpSENS\2';
        t = cputime;
        
        var_name = sprintf('CS_%d_.mat', j);
        COUNTER = COUNTER + 1;
        
        cs = struct(CS);
        data_struct = struct(data);
        sz_str = strcat('./COmpSENS/');
        save(fullfile(sz_str, var_name), 'cs', 'data_struct');
            
%         save(var_title,'statistics','data','max_projs','num_trials')
        t = cputime-t;
        disp(['Took ' num2str(t) ' seconds total.'])
%         clear 
end

