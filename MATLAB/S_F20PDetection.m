
%Provide path to folder (can have subfolders) wich contains your audio data - edit to execute
survey_list = dir('/Users/.../Documents/.../**/*.wav'); %Get file info from all wav files in folder and subfolders

%Define frequency band for fin whale pulses 
fb = [15,26];

%Define frequency bands for noise proportion of SNR 
nb = [10,15,30,80];

%Create empty table to store values and set counter to 0 to write into
%table
Tkurt = table();
c=0;

%Loop through wave files to compute metrics
for x = 1:length(survey_list)

    %Get sampling rate and duration info from wave file and store duration
    %in list
    inf = audioinfo([survey_list(x).folder '/' survey_list(x).name]);
    FS = inf.SampleRate;
    dur = inf.Duration;
    survey_list(x).Dur = dur;

    %Retrieve date and time info from filename to add to detection time
    dt = survey_list(x).name(1:15);
    if size(strsplit(dt,'-'),2) == 1
        dt = datetime(datenum(dt,'YYYYmmdd_hhMMss'),'ConvertFrom','datenum'); 
    else
        dt = datetime(datenum(dt,'YYYYmmdd-hhMMss'),'ConvertFrom','datenum'); 
    end
    dt.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';

    %Create bandpass filter according to sampling rate of file
    BP_filter = designfilt('bandpassiir','FilterOrder',20, ...
    'HalfPowerFrequency1',fb(1),'HalfPowerFrequency2',fb(2), ...
    'SampleRate',FS);

    %Read audiofile
    Y = audioread([survey_list(x).folder '/' survey_list(x).name]);  

    %Process audiofiles in 2s-sliding windows with 1.5s overlap
    for i = 0.5:0.5:dur-2
        
        %cut audio snippet
        Y_t = Y((i)*FS:(i+2)*FS-1);  

        %compute PSD of snippet
        [p,f] = pwelch(Y_t,hamming(FS),[],FS,FS);
        
        %Bandpass filter snippet
        Y_f = filter(BP_filter,Y_t);

        %Apply Teager-Kaiser-Energy Operator (TKEO)
        [ey,ex]=energyop(Y_f,false);
        
        %Retreive sample index of maximum TKEO value for timestamp of
        %detection
        [m,in] = max(ex);

        %Update counter and write filename and time of detection into table
        c=c+1;
        Tkurt.File{c} = survey_list(x).name;
        Tkurt.DateTime(c) = datenum(dt + seconds(i+(in/FS)));

        %Store kurtosis value and kurtosis product oin table
        Tkurt.Kurt(c) = kurtosis(Y_f);
        Tkurt.KurtProd(c) = kurtosis(Y_f) .* kurtosis(ex);
        
        %Calculate spectral SNR with frequency band limits and store into table
        Tkurt.SNRF(c) = (10*log10(mean(p(find(f >= fb(1) & f <= fb(2)),:)))) - (10*log10(mean(p(find(f >= nb(1) & f <= nb(2) | f >= nb(3) & f <= nb(4)),:))));
        
        %Split audio snippet into signal of 0.6s and noise before and after
        %(0.6s) depending on position of snippet within audio file
        if i <= 1.5
            Y_post = Y(ceil((i+2)*FS-1):ceil((i+2)*FS-1+2*FS));
            Y_t = [Y_t;Y_post];
            if in-round(0.6*FS) <= 0
            signal = Y_t(1 : in+round(0.6*FS)); 
            else
            signal = Y_t(in-round(0.6*FS) : in+round(0.6*FS));
            end
            noise = Y_t(in+round(0.6*FS) : in+round(1.2*FS));
        elseif i >= dur-3
            Y_pre = Y(floor((i)*FS-2*FS):floor((i)*FS));
            Y_t = [Y_pre;Y_t];
            in = length(Y_pre) + in;
            if in+round(0.6*FS) > length(Y_f)
            signal = Y_t(in-round(0.6*FS) : end);
            else
            signal = Y_t(in-round(0.6*FS) : in+round(0.6*FS));
            end
            noise = Y_t(in-round(1.2*FS) : in-round(0.6*FS));
        else
            Y_pre = Y(ceil((i)*FS-round(1.2*FS)):ceil((i)*FS));
            Y_post = Y(floor((i+2)*(FS-1)):floor((i+2)*(FS-1)+round(1.2*FS)));
            Y_t = [Y_pre;Y_t;Y_post];
            in = length(Y_pre) + in;
            signal = Y_t(in-round(0.6*FS) : in+round(0.6*FS));
            noise = Y_t([in-round(1.2*FS) : in-round(0.6*FS),in+round(0.6*FS) : in+round(1.2*FS)]);
        end
        
        %Calculate temporan SNR and store into table
        Tkurt.SNRT(c) = 20*log10(abs(rms(filter(BP_filter,signal))-rms(filter(BP_filter,noise)))/rms(filter(BP_filter,noise)));
        
        %Compute PSD of signal and noise and calculate the SNRs for
        %bandwidth
        [p,f] = pwelch(signal,hamming(FS),[],FS,FS);
        ps = p(f >= 13 & f <= 35); f = f(f >= 13 & f <= 35);
        [p,f] = pwelch(noise,hamming(FS),[],FS,FS);
        pn = p(f >= 13 & f <= 35); 
        BWSNR = 10*log10(ps)-10*log10(pn);
        
        %Store BW in table by counting the number of frequency bins where
        %SNR is above 0
        Tkurt.BW(c) = sum(BWSNR>=0);
    end
end

%set datetime format within table
Tkurt.DateTime = datetime(Tkurt.DateTime,'ConvertFrom','datenum');
Tkurt.DateTime.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';

%Join detections within 2s range by keeping maximum kurtosis, earliest
%time, and average SNRs
buff1 = 2; %in seconds 
for i = 1:height(Tkurt)
    ind = find(Tkurt.DateTime >= (Tkurt.DateTime(i)-seconds(buff1)) & Tkurt.DateTime <= (Tkurt.DateTime(i)+seconds(buff1)));
    if ind == i
       if i == height(Tkurt)
           break
       else
           continue
       end
    else
       Tkurt.Kurt(ind(1)) = max(Tkurt.Kurt(ind));
       Tkurt.KurtProd(ind(1)) = max(Tkurt.KurtProd(ind));
       Tkurt.SNRF(ind(1)) = max(Tkurt.SNRF(ind));
       Tkurt.SNRT(ind(1)) = max(Tkurt.SNRT(ind));
       Tkurt.BW(ind(1)) = max(Tkurt.BW(ind));
       Tkurt(ind(2:end),:) = [];
       if i == height(Tkurt)
          break
       end 
    end   
end

%Filter table for thresholds
% edit values in TH vector to filter for specific threshold values in the
% order TK1,TK2,TSF,TST,TST.2,TBW,TK1.2
TH = [2.5,40,8,-2,-8,75,4.75];
Tkurt_f = Tkurt(Tkurt.Kurt(:)>=TH(1) | Tkurt.KurtProd(:)>=TH(2),:);
Tkurt_f = Tkurt_f (Tkurt_f.SNRF(:)>=TH(3),:);
Tkurt_f = Tkurt_f(Tkurt_f.SNRT(:)>=TH(4) | Tkurt_f.SNRT(:)>=TH(5) & Tkurt_f.BW(:)>=(23/100*TH(6)) & Tkurt_f.Kurt(:)>=TH(7),:);

%Filter to only preserve only clusters of minimum 5 detections within 5
%minutes
qs = [];
for q = 1:height(Tkurt_f)
    if q > height(Tkurt_f)
      break
    end
      c5 = height(Tkurt_f(datenum(Tkurt_f.DateTime(:)) >= addtodate(datenum(Tkurt_f.DateTime(q)),-2.5*60,'second') & datenum(Tkurt_f.DateTime(:)) <= addtodate(datenum(Tkurt_f.DateTime(q)),2.5*60,'second'),:));
    if c5 < 5 
       qs = [qs,q];
    end
end
Tkurt_f(qs,:) = [];
            
%save table with computed values - edit path and filename to execute
writetable(Tkurt_f,['/Users/.../Documents/.../' 'FILENAME.csv'])
