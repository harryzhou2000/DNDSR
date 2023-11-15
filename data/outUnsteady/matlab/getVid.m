

prefix = ".."
namePrefix = "CylinderA1__"

files = dir(prefix);
fileNeed = strings(0);
fileNum = [];


for ifile = 1 : numel(files)
    file = files(ifile);
    match = regexp(file.name, prefix + "(\d+).png",'tokens');
    if numel(match) == 1

        fileNum(end + 1) = str2double(match{1});
        fileNeed(end + 1) = string(file.folder) + "/" + file.name;
    end

end

[fileNum, sortedI] = sort(fileNum);
fileNeed = fileNeed(sortedI);
%%

vidWriter = VideoWriter(prefix + "/" + namePrefix + ".mp4","MPEG-4");
vidWriter.Quality = 95;
vidWriter.FrameRate = 10;
vidWriter.open()
 
for ifile = 1 : numel(fileNeed)
    pic = imread(fileNeed(ifile));
    vidWriter.writeVideo(pic);
    fprintf("frame #%d\n", ifile);
end
%%
vidWriter.close()

