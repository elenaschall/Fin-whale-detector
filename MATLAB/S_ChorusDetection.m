
%Provide path to folder with wave files to process - edit path to execute
survey_list = dir('/Users/../Documents/.../*.wav'); %Get file info from all wav files in folder

%Create empty table to store values
FIAU = table('Size',[length(survey_list),10],'VariableTypes',{'string','double','double','double','double','double','double','double','double','double'});

%Define frequency bands for fin whale choruses 
fband = [17,25,84,87,96,100];   %LFC2 = 17-25Hz; HFC8 = 84-87Hz; HFC9 = 96-100Hz

%Define frequency bands for noise comparison 
nband = [11,15,30,33,81,83,90,94,103,105];  %noise below LFC2 = 11-15Hz; noise above LFC2 = 30-33Hz; noise below HFC8 = 81-83Hz; noise above HFC8 and below HFC9 = 90-94Hz; noise above HFC9 = 103-105Hz

%Loop through wave files to compute metrics
for i = 1:length(survey_list)

    %Get sampling rate and duration info from wave file
    inf = audioinfo([survey_list(i).folder '/' survey_list(i).name]);
    FS = inf.SampleRate;
    dur = inf.Duration;

    %Read and normalize wave file
    Y = audioread([survey_list(i).folder '/' survey_list(i).name]);  %Read audiofile at samples
    Y = normalize(Y);

    %If the sampling rate is higher than 500, the files get downsampled to
    %500
    if FS > 500
        Y = lowpass(Y,500/2,FS);
        [Numer, Denom] = rat(500/FS);
        Y = resample(Y,Numer,Denom);
        FS = 500;
    end

    %Calculate PSD (p) of wave file and extract frequency resolution (f)
    [p,f] = pwelch(Y,[],[],2048,FS);          
    
    %Calculate slope of PSD
    slo = gradient(p);
         
    %Extract average PSDs of chorus bands
    F1 = mean(p(find(f >= fband(1) & f <= fband(2)),:)); 
    F2 = mean(p(find(f >= fband(3) & f <= fband(4)),:)); 
    F3 = mean(p(find(f >= fband(5) & f <= fband(6)),:)); 

    %Extract median PSDs of noise bands
    N1 = median(p(find(f >= nband(1) & f <= nband(2) | f >= nband(3) & f <= nband(4)),:)); 
    N2 = median(p(find(f >= nband(5) & f <= nband(6) | f >= nband(7) & f <= nband(8)),:)); 
    N3 = median(p(find(f >= nband(7) & f <= nband(8) | f >= nband(9) & f <= nband(10)),:));

    %Extract area under PSD curve within chorus bands and noise bands
    a1 = trapz(f(find(f >= fband(1) & f <= fband(2))),p(find(f >= fband(1) & f <= fband(2)))) / trapz(f(find(f >= nband(1) & f <= nband(4))),p(find(f >= nband(1) & f <= nband(4))));
    a2 = trapz(f(find(f >= fband(3) & f <= fband(4))),p(find(f >= fband(3) & f <= fband(4)))) / trapz(f(find(f >= nband(5) & f <= nband(8))),p(find(f >= nband(5) & f <= nband(8))));
    a3 = trapz(f(find(f >= fband(5) & f <= fband(6))),p(find(f >= fband(5) & f <= fband(6)))) / trapz(f(find(f >= nband(7) & f <= nband(10))),p(find(f >= nband(7) & f <= nband(10))));
    
    %Extract slopes at borders of chorus
    s1 = max(slo(find(f>=fband(1)-2 & f<=fband(1)+2)));
    s2 = max(slo(find(f>=fband(3)-2 & f<=fband(3)+2))) - min(slo(find(f>=fband(4)-2 & f<=fband(4)+2)));
    s3 = max(slo(find(f>=fband(5)-2 & f<=fband(5)+2))) - min(slo(find(f>=fband(6)-2 & f<=fband(6)+2))); 
  
    %Compute SNRs and store all values in the prepared table
    FIAU(i,1) = {survey_list(i).name};
    FIAU{i,2} = (10*log10(F1)) - (10*log10(N1));   
    FIAU{i,3} = (10*log10(F2)) - (10*log10(N2));   
    FIAU{i,4} = (10*log10(F3)) - (10*log10(N3)); 
    FIAU{i,5} = a1;
    FIAU{i,6} = a2;
    FIAU{i,7} = a3;
    FIAU{i,8} = s1;
    FIAU{i,9} = s2;
    FIAU{i,10} = s3;
end

%set variable names of table with computed values with
% SNRL = SNR of LFC2; SNR8 = SNR of HFC8; SNR9 = SNR of HFC9
% AL = Area of LFC2; A8 = Area of HFC8; A9 = Area of HFC9
% SL = Slope of LFC2; S8 = Slope of HFC8; S9 = Slope of HFC9
FIAU.Properties.VariableNames = {'file','SNRL','SNR8','SNR9','AL','A8','A9','SL','S8','S9'};

%Filter table for thresholds
% edit values in TH vector to filter for specific threshold values in the
% order SNRL,SNR8,SNR9,AL,A8,A9,SL,S8,S9
TH = [4,0,0,0,0.3,0.35,0,0,0];
FIAU_f = FIAU(FIAU.SNRL >= TH(1) & FIAU.SNR8 >= TH(2) & FIAU.SNR9 >= TH(3) & FIAU.AL >= TH(4) & FIAU.A8 >= TH(5) & FIAU.A9 >= TH(6) & ...
    FIAU.SL >= TH(7) & FIAU.S8 >= TH(8) & FIAU.S9 >= TH(9),:);

%save table with computed values - edit path and filename to execute
writetable(FIAU_f,['/Users/.../Documents/.../' 'FILENAME.csv'])

