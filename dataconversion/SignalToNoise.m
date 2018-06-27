mainFolder = "Dataset";
subdirs = dir(mainFolder);

subdirs(~[subdirs.isdir]) = [];
%And this filters out the parent and current directory '.' and '..'
tf = ismember( {subdirs.name}, {'.', '..'});
subdirs(tf) = [];
numberOfFolders = length(subdirs);

subdirs = struct2cell(subdirs);
subdirs = subdirs(1, 1:numberOfFolders);

for innerIndex = 1:numberOfFolders
    audiofiles = dir(mainFolder + "/" + subdirs(innerIndex));
    %And this filters out the parent and current directory '.' and '..'
    tf = ismember( {audiofiles.name}, {'.', '..'});
    audiofiles(tf) = [];
    numberOfFiles = length(audiofiles);
    audiofiles = struct2cell(audiofiles);
    audiofiles = audiofiles(1, 1:numberOfFiles);
    disp("folder: " + subdirs(innerIndex))
    SNR = 0;
    for audiofile = audiofiles
        [y,Fs] = audioread(mainFolder + "/" + subdirs(innerIndex) + "/" + audiofile);
        yMono = (sum(y, 2) / size(y, 2));
        SNR = SNR + snr(yMono);
        disp(SNR)
    end
    disp("Total MEAN: " + num2str(SNR/numberOfFiles))
    disp(" ")
    
    
end