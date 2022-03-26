%% addpaths
addpath 'C:\Users\Dvir\Desktop\NV-centers Compressed learning\NV-centers Compressed learning\compressedLearning-main\L1_homotopy_v2.0'
%% Path to change and sim parameters to try
n = 0;
m = 0;

data_type = 'train';
num_of_measures = 60;
df=3;
% freqs_to_try = ones(1, n) * 3;%300 samples of 3 freqs together. Usages: 3 or [2,3,4];
freqs_to_try = 3;
base_noise = 0.0; %0.0004;%[0:0.0003:0.0009, 0.001:0.002:0.008];   
% base_noise = [0:0.0003:0.0009, 0.001:0.002:0.008];  
num_trials=1;

%% Get diamond simulation
for j = n : m
contrast = 0.002;
data = mock_diamond2_new(df);
% figure;plot(data.smp_freqs,data.target)  %Plots target data
target_num_pks = length(data.peak_locs); 
freqs = data.smp_freqs;
%file_title_sim = [location 'diamond' num2str(data.B_mag) 'G_' num2str(data.B_theta) 'theta_' num2str(data.B_phi) 'phi'];
%save(file_title_sim,'data');

window_start = min(data.peak_locs)-10;
window_end = max(data.peak_locs)+10;


%% initialize strucutre for saving data
% CS = spike_adaptive_l1homotopy(freqs, 1, ...
%     1, window_start, window_end, data);
% min_projs = CS.num_initial_projs;
% max_projs = CS.num_max_projs;
% meas_of_interest = min_projs:max_projs;

CS = Spike_adaptive_new(freqs, 1, ...
    1, window_start, window_end, num_of_measures, data);
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
%             CS = spike_adaptive_l1homotopy(data.smp_freqs, num_simulataneous_freqs, ...
%                 contrast, window_start, window_end, data);
              CS = Spike_adaptive_new(data.smp_freqs, num_simulataneous_freqs, ...
                contrast, window_start, window_end, num_of_measures, data);
            CS.display = false;
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
        
        

    end
    disp(['Done with noise=' num2str(curr_noise)])

end
     t = cputime;
                     
        cs = struct(CS);

        data_struct = struct();
        data_struct = setfield(data_struct, 'projs', cs.projs);
        data_struct = setfield(data_struct, 'curr_cs_per', data.curr_cs_per);
        data_struct = setfield(data_struct, 'magnetic_field', data.B_mag);
        data_struct = setfield(data_struct, 'measures', cs.sig_measurements);
        data_struct = setfield(data_struct, 'smp_freqs', cs.smp_freqs);
        data_struct = setfield(data_struct, 'peak_locs', data.peak_locs);
        data_struct = setfield(data_struct, 'B_projs', data.B_projs);
        data_struct = setfield(data_struct, 'B_vec', data.B_vec);
        data_struct = setfield(data_struct, 'target', data.target);
        data_struct = setfield(data_struct, 'df', df);
            
        switch data_type
            case 'train' 
                var_name = sprintf('CS_%d_.mat', j);
            case 'validation' 
                var_name = sprintf('valid_CS_%d_.mat', j);
            case 'testing' 
                var_name = sprintf('test_CS_%d_.mat', j);
        
        end
        
        % PAY ATTENTION TO THE NUMBER SAMPS! DO NOT OVERRIDE!!
        

        sz_str = strcat('./COmpSENS/fixed_number_of_measures/df_3/60_measures/3_simultaneous_freqs/', data_type, '/');
%         sz_str = strcat('./COmpSENS/');
        
        save(fullfile(sz_str, var_name), 'data_struct');
%         save('data_struct', 'data _struct')

        % sz_str = strcat('./COmpSENS');
%         save(fullfile(sz_str, var_name), 'cs', 'data_struct');
%         save(fullfile(sz_str, var_name_valid), 'cs', 'data_struct');
            
%         save(var_title,'statistics','data','max_projs','num_trials')
        t = cputime-t;
        disp(['Took ' num2str(t) ' seconds total.'])
%         clear 
end

