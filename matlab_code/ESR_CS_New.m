classdef ESR_CS_New < ExpClass
    %Runs Compressive Senssing ESR, using either TVAL3 or L1. works with 1-4 frequencies
    %make sure TVAL or L1 are available on path
    
    properties
        frequency%in MHz
        amplitude
        averaged_meas
        averaged_ref
        averaged_devision
        averaged_ref_rep
        averaged_meas_rep
        averaged_devision_rep
        mode % cw or pulsed
        numChannels %
        normalized
        contrast
        freq_range
        I_helm
        CS
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        function obj = ESR_CS_New % Defult values go here **works with spike_adaptive)l1homotopy
            obj@ExpClass();
            obj.repeats =1;  %% If changing repeats, must go to cameraClass and change 'FrameCount' on line 86 to be repeats*2
            obj.detectionDuration = 10000;% us %Same as the exposure time
            obj.averages =1;
            %obj.frequency = [2791.2,2796.6,2811,2815,2823.6,2829,2842.8,2847.8,2893.4,2896.8,2910.8,2914.8,2921.2,2927.2,2938.4,2943.8];
            %obj.frequency = linspace(2750,2970,51);
            obj.amplitude = [-10,-10,-10,-30]; %dBm
            obj.mode = 'CW'; % cw or pulsed (to be implemented in the future - if needed
            obj.numChannels = 3; % 4 channels can be added.
            obj.contrast= 0;
            obj.freq_range=linspace(2661,3100,200);
            obj.I_helm = [0.3, 1.6 ,4.2];
        end
        
        function set.frequency(obj,newVal)
%             OK1 = obj.TestMaxLength(newVal,1e4);
            OK2 = obj.TestLim(newVal,0,6e3);%limits in  MHz
            if  OK2
                obj.frequency = newVal;
                obj.changeFlag = 1;
            end
        end
        function set.amplitude(obj,newVal)
            %OK1 = obj.TestMaxLength(newVal,length(obj.FG));
            OK2 = obj.TestLim(newVal,-60,10);%limits in  MHz
            if  OK2
                obj.amplitude = newVal;
                obj.changeFlag = 1;
            end
        end
        function set.numChannels(obj,newVal)
           % OK1 = obj.TestMaxLength(newVal,1);
            %OK2 = obj.TestLim(newVal,1,length(obj.FG));%limits in  MHz
            OK3 = obj.TestScalar(newVal);
            if   OK3
                obj.numChannels = newVal;
                obj.changeFlag = 1;
            end
        end
        
        function set.mode(obj,newVal)
            switch lower(newVal)
                case{'cw','continuous','cont','cwesr'}
                    obj.mode = 'CW';
                case{'pulsed','pulse','pulsedesr'}
                    obj.mode = 'pulsed';
                otherwise
                    warning('Unknown option')
            end
            obj.changeFlag = 1;
        end
        
        
        function exp = SaveExperiment(obj,fileName)
            % Save all properties as exp.propertyName, and adds a exp.date
            % field. If file name is given - sames using filename. Elese -
            % open a GUI interface
            % saves using
            names = fields(obj);
            for k = 1:length(names)
                command = sprintf('exp.%s = obj.%s;',names{k},names{k});
                eval(command);
            end
            exp.date = datestr(now,'yyyy-mm-dd_HH-MM-SS');
            eval(sprintf('save(''%s'',''exp'');',fileName));
        end
        
        %%%%%
        function LoadExperiment(obj) %X - detection times, in us
            %             cameraDelayVal=max(10^3,(-9.5*exp(-1.6*obj.camera.ROI/10^6)+10.17)*2*10^3); %in us
            cameraDelayVal = 2*max(10^3,(-9.5*exp(-1.6*obj.camera.ROI/10^6)+10.17)*2*10^3); %in us
            
            obj.PB.setRepeats(obj.repeats);
            %%% load the experiment
            obj.PB.newSequence;
            if length(obj.detectionDuration) ~= 1
                error('a single detection duration is used in cw mode')
            end
            obj.PB.newSequenceLine({},max(1e4,cameraDelayVal)) % This is because of a weird PB bug which causes glitch pulses at the beginning.
            obj.PB.newSequenceLine({'greenLaser'},10)
            obj.PB.newSequenceLine({'greenLaser','MW','detector'},obj.detectionDuration, 'reference')
            obj.PB.newSequenceLine({},cameraDelayVal, 'cameraDelay') %depends on ROI that changes the maximum fps. Determined by calibration, can be further optimized for large ROIs.
            obj.PB.newSequenceLine({'greenLaser'},10)
            obj.PB.newSequenceLine({'greenLaser','detector'},obj.detectionDuration)
            obj.changeFlag = 0;
            for i=1:obj.numChannels
                if mod(i,2)==0
                    chan='B';
                else
                    chan='A';
                end
                index=ceil(i/2);
                obj.EnableOutput(1,chan,index)
            end
        end
        
        function Run(obj) %X is the MW duration vector
            obj.LoadExperiment;
            obj.stopFlag = 0;
            obj.SetAmplitude(obj.amplitude(1),1,1); %SetAmplitude(obj,val,channel,index)
            if obj.numChannels >1
                if length(obj.amplitude)<length(obj.numChannels)
                    error('amplitude must have a value for the second MW channel')
                end
                obj.SetAmplitude(obj.amplitude(2),2,1) %format- SetAmplitude(obj,val,channel,index)
                obj.SetAmplitude(obj.amplitude(3),1,2)
                obj.SetAmplitude(obj.amplitude(4),2,2)
            end
            %[window_start, window_end, peaks_locations] =Initial_parameters(obj.I_helm);
            window_start=2730;
            window_end= 3012;
            %  obj.CS = adaptive_reconstruction_full_data(obj.freq_range,obj.numChannels,obj.contrast);
            %             obj.CS = adaptive_reconstruction_NEW(obj.freq_range,obj.numChannels,obj.contrast,window_start,window_end);
            obj.CS = spike_adaptive_l1homotopy(obj.freq_range,obj.numChannels,obj.contrast,window_start,window_end);
%            obj.CS = spike_adaptive_TVAL3(obj.freq_range,obj.numChannels,obj.contrast,window_start,window_end); 
           n = 2* obj.repeats;%%%%%%%%%%%%%%
            
            
            x=obj.camera.x;
            y=obj.camera.y;
            obj.signal = zeros(y,x,2,1); %third dimention- 2x(meas+ref)  % The 1 will grow
            obj.frequency = [];
            try
                k=0;
                while obj.CS.Continue %CS
                    obj.averaged_ref_rep=zeros(y,x);
                    obj.averaged_meas_rep=zeros(y,x);
                    k=k+1;
                    success=false;

                        [obj.CS, freq] = obj.CS.frequency();
                        obj.frequency = [obj.frequency; freq];
                    for trial = 1 : 3
                        try
                            
                            if obj.numChannels==1
                                obj.SetFrequency(freq,1,'A'); %%CS  SetFrequency(obj,val,index, channel)
                            elseif obj.numChannels==2
%                                 obj.SetFrequency(freq(2),1,'B');
%                                 obj.SetFrequency(freq(1),1,'A');
                                obj.SetFrequency(freq(2),1,'A');
                                obj.SetFrequency(freq(1),1,'B'); %galya
                            elseif obj.numChannels==3
                                obj.SetFrequency(freq(2),1,'A');
                                obj.SetFrequency(freq(1),2,'A');
                                obj.SetFrequency(freq(3),1,'B');
                            elseif obj.numChannels==4
                                obj.SetFrequency(freq(2),1,'B');
                                obj.SetFrequency(freq(1),1,'A');
                                obj.SetFrequency(freq(4),2,'B');
                                obj.SetFrequency(freq(3),2,'A');
                            end
                            
                            
                            obj.camera.PrepareRead(n); %1 -Just Measurement. 2- Meas and Ref
                            obj.PB.Run;
                            I = obj.camera.Read(n, obj.detectionDuration/100); %timeout for the camera, when wating for image.
                            obj.camera.stopRead();
                            success = true;
                            break
                        catch err
                            warning(err.message);
                            fprintf('Experiment failed at trial %d, attempting again.\n',trial);
                            obj.camera.stopRead();
                        end
                    end
                    I_meas = mean(I(:,:,1:2:end),3);
                    I_ref = mean(I(:,:,2:2:end),3);
                    obj.signal(:,:,1,k) = I_meas; %meas
                    obj.signal(:,:,2,k) = I_ref; %ref
                    
                    
                    if ~success
                        break
                    end
                    
                    obj.CS = obj.CS.process_data(obj.signal(:,:,:,k));
                    disp(k)
                    drawnow
                    if obj.stopFlag == 1
                        break;
                    end
                    
                end
                obj.CloseExperiment()
            catch err
                try
                    obj.CloseExperiment()
                catch err2
                    warning(err2.message)
                end
                rethrow(err)
            end
        end
       
        
        function PlotResults(obj,index)
            if isempty(obj.figHandle) || ~isvalid(obj.figHandle)
                %figure;
                obj.figHandle = gca;
                %obj.gui.stopButton = uicontrol('Parent',gcf,'Style','pushbutton','String','Stop','Position',[0.0 0.5 100 20],'Visible','on','Callback',@obj.PushBottonCallback);
            end
            averaged_meas2 = obj.averaged_meas;
            averaged_ref2 = obj.averaged_ref;
            averaged_dev2=obj.averaged_devision ;
            freq=obj.frequency;
            pix=squeeze(mean(mean(averaged_meas2,1),2));
            pix2=squeeze(mean(mean(averaged_ref2,1),2));
            pix3=squeeze(mean(mean(averaged_dev2,1),2));
            figure(6)
            plot(freq,pix./pix2)
            xlabel('Frequency (MHz)')
            ylabel('FL (norm)')
            figure(7)
            plot(freq,pix,freq,pix2)
            %             figure(8)
            %             plot(freq,pix3)
            %             title('devision')
        end
        
        function PushBottonCallback(obj,PushButton, EventData)
            obj.stopFlag = 1;
        end
    end
end

