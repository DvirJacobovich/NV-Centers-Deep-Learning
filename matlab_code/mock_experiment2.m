% Experiment Simulation
% how many frequenciese send at once
num_simulataneous_freqs = 5; % 2
contrast = 0.005;

data = mock_diamond2;
data.sigma = 0.000;%0.00025; % if you want to play with the noise change here
freqs = data.smp_freqs;

window_start = min(data.peak_locs)-20;
window_end = max(data.peak_locs)+20;
CS = spike_adaptive_l1homotopy(freqs, num_simulataneous_freqs, ...
    contrast, window_start, window_end,data);
% CS = spike_adaptive_NLLS(freqs, num_simulataneous_freqs, ...
%     contrast, window_start, window_end);
% 
%  CS = spike_adaptive_reconstruction(freqs, num_simulataneous_freqs, ...
%      contrast, window_start, window_end);

tic
while CS.Continue
    [CS,current_freq] = CS.frequency();
    data = data.getMeasurement(current_freq);
    CS = CS.process_data(data.signal);
end
toc
disp(mean(abs(data.peak_locs-CS.pk_locs)))

[nrows,ncols,~] = size(data.sig);

target_data = data.getRaster(freqs);
% if CS.est_mean~= 1
%     target_data =(ref-sig)./(mean(ref));
% else
%     target_data =(ref-sig)./(mean(ref));
% end

if CS.display
    target_data = target_data-mean(target_data);
    hold on
    plot(CS.ax,CS.smp_freqs,target_data)
    plot(CS.ax,data.smp_freqs,data.target-mean(data.target))
    legend('Reconstruction','Raster Estimation','Ground Truth')
    
    if CS.get_err
        grid on
        xlabel('MHz')
        ylabel('Zero Mean Absorption')
        title([num2str(num_simulataneous_freqs) ' frequencies, \sigma='...
            num2str(data.sigma)])
        hold off
        if ~isempty(CS.actual_err)
            figure
            plot(CS.meas_num,CS.actual_err)
            grid on
            xlabel('MHz')
            ylabel('Mean Error (MHz)')
            title('Single Simulation')
        end
    end
end

plot_derivatives = false;
if plot_derivatives
    [~, curr_num_pks, curr_guess] = getFitGuess(CS.smp_freqs,...
        target_data, 0.004);
    [raster_fit, ~] = ...
        lorentzian_fit(CS.smp_freqs',...
        target_data', 2, 2, curr_num_pks, curr_guess);    
    figure
    hold on
    slope_recon = diff(CS.fit-mean(CS.fit));
    slope_raster = diff(raster_fit-mean(raster_fit));
    slope_target = diff(data.target-mean(data.target));
    deriv_freqs = CS.smp_freqs(1:(end-1));
    df = 2;
    plot(deriv_freqs,slope_recon/df)
    plot(deriv_freqs,slope_raster/df)
    plot(deriv_freqs,slope_target/df)
    grid on
    xlabel('Frequency (MHz)')
    ylabel('Slope (MHz^{-1})')
    title_str = ['Slope Determination, ' num2str(data.B_mag)...
        ' G, \theta=' num2str(data.B_theta) char(176) ', \phi=' ...
        num2str(data.B_phi) char(176)];
    title(title_str)
    legend('Reconstruction','Raster Estimation','Ground Truth')    
end
