%% Basic example showing samples from the dataset
clear, close all

% chose set
set = 'training';
% set = 'evaluation';

% load annotations
annotations = load(sprintf('./%s/anno_%s.mat', set, set));

% iterate samples
for field=fieldnames(annotations)'
    frame_id = str2num(field{1}(6:end));
    
    % load data
    img = imread(sprintf('./%s/color/%.5d.png', set, frame_id));
    depth = imread(sprintf('./%s/depth/%.5d.png', set, frame_id));
    mask = imread(sprintf('./%s/mask/%.5d.png', set, frame_id));
    
    % convert depth
    depth = uint16(depth(:, :, 2)) + bitsll(uint16(depth(:, :, 1)), 8);
    depth = 5.0 / (2^16-1) * double(depth);
    
    % load from annotations
    anno = getfield(annotations, field{1});
    coord_uv = anno.uv_vis(:, 1:2);
    coord_visible = anno.uv_vis(:, 3) == 1;
    coord_xyz = anno.xyz;
    K = anno.K;
    
    % project world coords into camera frame
    coord_uv_proj = coord_xyz * K';
%     coord_uv_proj = coord_uv_proj(:, 1:2) ./ coord_uv_proj(:, 3);  % this line doesn't work for some Matlab versions
    coord_uv_proj = bsxfun(@rdivide, coord_uv_proj(:, 1:2), coord_uv_proj(:, 3));
    
    % visualize
    fh = figure(1);
    subplot(2,2,1), imshow(img)
    hold all, plot(coord_uv(coord_visible, 1), coord_uv(coord_visible, 2), 'go'), hold off
    hold all, plot(coord_uv_proj(coord_visible, 1), coord_uv_proj(coord_visible, 2), 'r+'), hold off
    subplot(2,2,2), imagesc(depth)
    subplot(2,2,3), imagesc(mask)
    subplot(2,2,4), scatter3(coord_xyz(coord_visible, 1), coord_xyz(coord_visible, 2), coord_xyz(coord_visible, 3)), view(0.0, -90.0)
    waitfor(fh);
end


