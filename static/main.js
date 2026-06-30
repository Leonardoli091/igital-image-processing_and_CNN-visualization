const MODEL_SCENES = {
  vgg16: [
    { type: "image", title: "输入图像", dims: "[3,224,224]" },
    { type: "kernel", title: "conv1_1", kernel: "[64,3,3,3]", out: "64个卷积核" },
    { type: "feature", layer: "conv1_2", title: "conv1输出", dims: "[64,224,224]", size: "large", depth: 4 },
    { type: "pool", title: "maxpool", dims: "[64,112,112]" },
    { type: "kernel", title: "conv2", kernel: "[128,64,3,3]", out: "128个卷积核" },
    { type: "feature", layer: "conv2_2", title: "conv2输出", dims: "[128,112,112]", size: "mid", depth: 5 },
    { type: "pool", title: "maxpool", dims: "[128,56,56]" },
    { type: "kernel", title: "conv3", kernel: "[256,128,3,3]", out: "256个卷积核" },
    { type: "feature", layer: "conv3_3", title: "conv3输出", dims: "[256,56,56]", size: "mid", depth: 6 },
    { type: "pool", title: "maxpool", dims: "[256,28,28]" },
    { type: "kernel", title: "conv4", kernel: "[512,256,3,3]", out: "512个卷积核" },
    { type: "feature", layer: "conv4_3", title: "conv4输出", dims: "[512,28,28]", size: "small", depth: 7 },
    { type: "pool", title: "maxpool", dims: "[512,14,14]" },
    { type: "kernel", title: "conv5", kernel: "[512,512,3,3]", out: "512个卷积核" },
    { type: "feature", layer: "conv5_3", title: "conv5输出", dims: "[512,14,14]", size: "tiny", depth: 7 },
    { type: "fc", title: "分类层", dims: "1 x 1 x 4096" },
  ],
  resnet18: [
    { type: "image", title: "原始图片", dims: "[3,224,224]" },
    { type: "kernel", title: "conv1", kernel: "[64,3,7,7]", out: "stride=2" },
    { type: "feature", layer: "conv1", title: "conv1输出", dims: "[64,112,112]", size: "large", depth: 4 },
    { type: "pool", title: "maxpool", dims: "[64,56,56]" },

    { type: "kernel", title: "layer1 block1 conv1", kernel: "[64,64,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer1", title: "block1 conv1输出", dims: "[64,56,56]", size: "mid", depth: 4 },
    { type: "kernel", title: "layer1 block1 conv2", kernel: "[64,64,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer1", title: "block1 conv2输出", dims: "[64,56,56]", size: "mid", depth: 4 },
    { type: "add", title: "残差相加", dims: "[64,56,56]" },
    { type: "kernel", title: "layer1 block2 conv1", kernel: "[64,64,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer1", title: "block2 conv1输出", dims: "[64,56,56]", size: "mid", depth: 4 },
    { type: "kernel", title: "layer1 block2 conv2", kernel: "[64,64,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer1", title: "block2 conv2输出", dims: "[64,56,56]", size: "mid", depth: 4 },
    { type: "add", title: "残差相加", dims: "[64,56,56]" },
    { type: "feature", layer: "layer1", title: "layer1最终输出", dims: "[64,56,56]", size: "mid", depth: 4 },

    { type: "kernel", title: "layer2 block1 conv1", kernel: "[128,64,3,3]", out: "stride=2" },
    { type: "feature", layer: "layer2", title: "block1 conv1输出", dims: "[128,28,28]", size: "small", depth: 5 },
    { type: "kernel", title: "layer2 block1 conv2", kernel: "[128,128,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer2", title: "block1 conv2输出", dims: "[128,28,28]", size: "small", depth: 5 },
    { type: "kernel downsample", title: "downsample", kernel: "[128,64,1,1]", out: "stride=2" },
    { type: "add", title: "残差相加", dims: "[128,28,28]" },
    { type: "kernel", title: "layer2 block2 conv1", kernel: "[128,128,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer2", title: "block2 conv1输出", dims: "[128,28,28]", size: "small", depth: 5 },
    { type: "kernel", title: "layer2 block2 conv2", kernel: "[128,128,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer2", title: "block2 conv2输出", dims: "[128,28,28]", size: "small", depth: 5 },
    { type: "add", title: "残差相加", dims: "[128,28,28]" },
    { type: "feature", layer: "layer2", title: "layer2最终输出", dims: "[128,28,28]", size: "small", depth: 5 },

    { type: "kernel", title: "layer3 block1 conv1", kernel: "[256,128,3,3]", out: "stride=2" },
    { type: "feature", layer: "layer3", title: "block1 conv1输出", dims: "[256,14,14]", size: "tiny", depth: 6 },
    { type: "kernel", title: "layer3 block1 conv2", kernel: "[256,256,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer3", title: "block1 conv2输出", dims: "[256,14,14]", size: "tiny", depth: 6 },
    { type: "kernel downsample", title: "downsample", kernel: "[256,128,1,1]", out: "stride=2" },
    { type: "add", title: "残差相加", dims: "[256,14,14]" },
    { type: "kernel", title: "layer3 block2 conv1", kernel: "[256,256,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer3", title: "block2 conv1输出", dims: "[256,14,14]", size: "tiny", depth: 6 },
    { type: "kernel", title: "layer3 block2 conv2", kernel: "[256,256,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer3", title: "block2 conv2输出", dims: "[256,14,14]", size: "tiny", depth: 6 },
    { type: "add", title: "残差相加", dims: "[256,14,14]" },
    { type: "feature", layer: "layer3", title: "layer3最终输出", dims: "[256,14,14]", size: "tiny", depth: 6 },

    { type: "kernel", title: "layer4 block1 conv1", kernel: "[512,256,3,3]", out: "stride=2" },
    { type: "feature", layer: "layer4", title: "block1 conv1输出", dims: "[512,7,7]", size: "micro", depth: 7 },
    { type: "kernel", title: "layer4 block1 conv2", kernel: "[512,512,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer4", title: "block1 conv2输出", dims: "[512,7,7]", size: "micro", depth: 7 },
    { type: "kernel downsample", title: "downsample", kernel: "[512,256,1,1]", out: "stride=2" },
    { type: "add", title: "残差相加", dims: "[512,7,7]" },
    { type: "kernel", title: "layer4 block2 conv1", kernel: "[512,512,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer4", title: "block2 conv1输出", dims: "[512,7,7]", size: "micro", depth: 7 },
    { type: "kernel", title: "layer4 block2 conv2", kernel: "[512,512,3,3]", out: "stride=1" },
    { type: "feature", layer: "layer4", title: "block2 conv2输出", dims: "[512,7,7]", size: "micro", depth: 7 },
    { type: "add", title: "残差相加", dims: "[512,7,7]" },
    { type: "feature", layer: "layer4", title: "layer4最终输出", dims: "[512,7,7]", size: "micro", depth: 7 },

    { type: "pool", title: "avgpool", dims: "[512,1,1]" },
    { type: "fc", title: "fc", dims: "[1000]" },
  ],
};

const state = {
  uploadedFile: null,
  selectedExample: "",
  selectedImageLabel: "",
  examples: [],
  cnnLayers: {},
  classicalOperations: [],
  cnnPreprocessedFile: null,
  cnnPreprocessedUrl: "",
  cnnPreprocessLabel: "",
  cnnPreprocessOperation: "__original__",
  diagramStage: "stem",
};

const CLASSICAL_OPERATION_FALLBACKS = {
  grayscale: {
    category: "灰度与点运算",
    principle: "按人眼亮度敏感度对 RGB 三通道加权，把彩色图像转换成单通道亮度图。",
    principle_points: ["公式：Y = 0.299R + 0.587G + 0.114B，绿色权重最大。", "输出只保留明暗结构，常作为边缘检测和阈值分割的前处理。"],
  },
  invert: {
    category: "灰度与点运算",
    principle: "对每个像素做反相映射，使亮区域变暗、暗区域变亮。",
    principle_points: ["公式：out = 255 - in。", "边界位置不变，只改变颜色和灰度方向。"],
  },
  brightness_contrast: {
    category: "灰度增强",
    principle: "使用线性灰度变换同时调整整体亮度和像素差异。",
    principle_points: ["本项目公式：out = 1.18 * in + 16。", "系数 1.18 放大对比度，偏置 16 提升整体亮度。"],
  },
  contrast_stretch: {
    category: "灰度增强",
    principle: "把主体灰度区间重新拉伸到完整 0 到 255 范围。",
    principle_points: ["取 2% 和 98% 分位作为低、高端点。", "避开少量极端像素，让主体灰度层次更明显。"],
  },
  gamma_correction: {
    category: "灰度增强",
    principle: "用幂函数进行非线性亮度校正。",
    principle_points: ["公式：out = 255 * (in / 255)^gamma，本项目 gamma = 0.65。", "gamma 小于 1 会明显提升暗部细节。"],
  },
  log_transform: {
    category: "灰度增强",
    principle: "用对数函数压缩高亮区域，同时拉开暗部灰度差异。",
    principle_points: ["公式：out = log(1 + in) / log(256) * 255。", "低灰度段增长快，高灰度段增长慢。"],
  },
  hist_equalization: {
    category: "灰度增强",
    principle: "根据灰度直方图的累计分布函数重新分配灰度级。",
    principle_points: ["先统计直方图，再计算 CDF。", "用 CDF 映射旧灰度，使灰度分布更均匀。"],
  },
  rgb_hist_equalization: {
    category: "灰度增强",
    principle: "只增强彩色图像的亮度信息，尽量保持原有色彩比例。",
    principle_points: ["先估计亮度并对亮度做均衡。", "再按比例缩放 RGB，减少直接均衡三通道造成的偏色。"],
  },
  mean_denoise: {
    category: "平滑去噪",
    principle: "用邻域平均值替代中心像素，削弱随机噪声。",
    principle_points: ["本项目使用 3x3 均值卷积核，每个权重都是 1/9。", "噪声会被平均掉，但边缘也会被一起模糊。"],
  },
  median_denoise: {
    category: "平滑去噪",
    principle: "用邻域中位数替代中心像素，适合去除孤立极值噪声。",
    principle_points: ["在 3x3 邻域内排序，取中间值输出。", "椒盐噪声常位于排序两端，会被中位数替换。"],
  },
  gaussian_denoise: {
    category: "平滑去噪",
    principle: "按高斯权重对邻域加权平均，实现更自然的平滑。",
    principle_points: ["本项目使用 5x5 高斯核，中心权重大、远处权重小。", "相比均值滤波，对边缘破坏通常更小。"],
  },
  motion_blur: {
    category: "平滑去噪",
    principle: "沿固定方向对像素做平均，模拟运动产生的拖影。",
    principle_points: ["本项目使用 9x9 对角线平均核。", "只有对角方向有权重，所以模糊具有方向性。"],
  },
  sharpen: {
    category: "锐化与边缘检测",
    principle: "增强中心像素并减去邻域像素，使灰度突变更明显。",
    principle_points: ["卷积核：[[0,-1,0],[-1,5,-1],[0,-1,0]]。", "平坦区域变化小，边缘和纹理会被强化。"],
  },
  unsharp_mask: {
    category: "锐化与边缘检测",
    principle: "先提取高频细节，再把细节叠加回原图。",
    principle_points: ["先做高斯模糊，再计算 detail = 原图 - 模糊图。", "输出 = 原图 + detail，用于增强清晰度。"],
  },
  emboss: {
    category: "锐化与边缘检测",
    principle: "用方向性正负卷积核把灰度变化转换成明暗阴影。",
    principle_points: ["一侧像素被加亮，另一侧像素被压暗。", "边缘会呈现类似光照下凸起的浮雕效果。"],
  },
  sobel: {
    category: "锐化与边缘检测",
    principle: "分别计算水平和垂直一阶梯度，再合成边缘强度。",
    principle_points: ["公式：magnitude = sqrt(Gx^2 + Gy^2)。", "Sobel 中间行或列权重为 2，带有轻微平滑作用。"],
  },
  prewitt: {
    category: "锐化与边缘检测",
    principle: "使用 Prewitt 模板估计水平和垂直方向的一阶梯度。",
    principle_points: ["Gx 和 Gy 每行或每列权重相同。", "计算简单，适合展示基础梯度边缘检测。"],
  },
  roberts: {
    category: "锐化与边缘检测",
    principle: "使用 2x2 交叉差分检测对角方向的灰度突变。",
    principle_points: ["计算两个对角方向的灰度差。", "对细小边缘敏感，但也更容易受噪声影响。"],
  },
  laplacian: {
    category: "锐化与边缘检测",
    principle: "使用二阶差分检测灰度变化率突变的位置。",
    principle_points: ["卷积核：[[0,-1,0],[-1,4,-1],[0,-1,0]]。", "不区分具体方向，多方向边缘都会响应。"],
  },
  high_pass: {
    category: "锐化与边缘检测",
    principle: "抑制低频平滑背景，保留高频边缘和纹理。",
    principle_points: ["卷积核：中心为 8，周围 8 个位置为 -1。", "中心像素与周围邻域作差，平滑区域会被抵消。"],
  },
  otsu_threshold: {
    category: "阈值分割",
    principle: "自动寻找能最好分开前景和背景的全局阈值。",
    principle_points: ["遍历所有候选阈值 T，计算前景和背景的类间方差。", "选择类间方差最大的 T，然后输出黑白二值图。"],
  },
  adaptive_threshold: {
    category: "阈值分割",
    principle: "每个局部区域单独计算阈值，适合光照不均图像。",
    principle_points: ["本项目使用 21x21 局部均值作为基础阈值。", "判断规则：gray >= local_mean - 5 时输出白色。"],
  },
  morph_erode: {
    category: "形态学处理",
    principle: "腐蚀会收缩二值图中的白色前景区域。",
    principle_points: ["先用 Otsu 得到二值图，再用 3x3 结构元素扫描。", "只有邻域满足前景条件才保留白色，所以小白点会被去除。"],
  },
  morph_dilate: {
    category: "形态学处理",
    principle: "膨胀会扩张二值图中的白色前景区域。",
    principle_points: ["先用 Otsu 得到二值图，再用 3x3 结构元素扫描。", "只要邻域中有前景就输出白色，可连接细小断裂。"],
  },
  morph_open: {
    category: "形态学处理",
    principle: "开运算是先腐蚀后膨胀，主要用于去除小前景噪声。",
    principle_points: ["腐蚀先删除小白点。", "膨胀再尽量恢复较大目标的尺寸。"],
  },
  morph_close: {
    category: "形态学处理",
    principle: "闭运算是先膨胀后腐蚀，主要用于填补小孔洞和缝隙。",
    principle_points: ["膨胀先连接断裂并填充小黑洞。", "腐蚀再尽量恢复目标边界大小。"],
  },
  posterize: {
    category: "色彩与风格化",
    principle: "把连续颜色压缩成少量离散色阶，形成海报化效果。",
    principle_points: ["本项目每个颜色通道压缩为 4 个等级。", "先按 step = 256 / 4 分桶，再取每个桶的中心值。"],
  },
};

const RESNET_STAGE_TABS = [
  { id: "stem", label: "Stem", layer: "conv1" },
  { id: "layer1", label: "layer1", layer: "layer1" },
  { id: "layer2", label: "layer2", layer: "layer2" },
  { id: "layer3", label: "layer3", layer: "layer3" },
  { id: "layer4", label: "layer4", layer: "layer4" },
  { id: "head", label: "分类头", layer: "layer4" },
];

const RESNET_STAGE_SCENES = {
  stem: {
    title: "Stem（输入层）",
    subtitle: "输入图像先经过 7x7 卷积，再经过 BN、ReLU 和 maxpool，尺寸从 224x224 降到 56x56。",
    nodes: [
      { type: "image", title: "输入图像", dims: "[3,224,224]" },
      { type: "kernel", title: "conv1", kernel: "[64,3,7,7]", out: "stride=2, padding=3" },
      { type: "feature", layer: "conv1", title: "conv1 输出", dims: "[64,112,112]", size: "large", depth: 4 },
      { type: "kernel", title: "bn1 + relu", kernel: "尺寸不变", out: "激活" },
      { type: "feature", layer: "conv1", title: "激活后", dims: "[64,112,112]", size: "large", depth: 4 },
      { type: "pool", title: "maxpool", dims: "[64,56,56]" },
    ],
  },
  layer1: {
    title: "layer1（2 个 BasicBlock）",
    subtitle: "通道保持 64，空间尺寸保持 56x56；两个 block 都是直接残差相加。",
    nodes: [
      { type: "feature", layer: "layer1", title: "输入", dims: "[64,56,56]", size: "mid", depth: 4 },
      { type: "kernel", title: "block1 conv1", kernel: "[64,64,3,3]", out: "stride=1" },
      { type: "kernel", title: "block1 conv2", kernel: "[64,64,3,3]", out: "stride=1" },
      { type: "add", title: "block1 残差相加", dims: "[64,56,56]" },
      { type: "kernel", title: "block2 conv1", kernel: "[64,64,3,3]", out: "stride=1" },
      { type: "kernel", title: "block2 conv2", kernel: "[64,64,3,3]", out: "stride=1" },
      { type: "add", title: "block2 残差相加", dims: "[64,56,56]" },
      { type: "feature", layer: "layer1", title: "layer1 输出", dims: "[64,56,56]", size: "mid", depth: 4 },
    ],
  },
  layer2: {
    title: "layer2（2 个 BasicBlock）",
    subtitle: "第一个 block 下采样，通道 64 -> 128，尺寸 56x56 -> 28x28。",
    nodes: [
      { type: "feature", layer: "layer2", title: "输入", dims: "[64,56,56]", size: "mid", depth: 4 },
      { type: "kernel", title: "block1 conv1", kernel: "[128,64,3,3]", out: "stride=2" },
      { type: "kernel", title: "block1 conv2", kernel: "[128,128,3,3]", out: "stride=1" },
      { type: "kernel downsample", title: "downsample", kernel: "[128,64,1,1]", out: "stride=2" },
      { type: "add", title: "block1 残差相加", dims: "[128,28,28]" },
      { type: "kernel", title: "block2 conv1", kernel: "[128,128,3,3]", out: "stride=1" },
      { type: "kernel", title: "block2 conv2", kernel: "[128,128,3,3]", out: "stride=1" },
      { type: "add", title: "block2 残差相加", dims: "[128,28,28]" },
      { type: "feature", layer: "layer2", title: "layer2 输出", dims: "[128,28,28]", size: "small", depth: 5 },
    ],
  },
  layer3: {
    title: "layer3（2 个 BasicBlock）",
    subtitle: "第一个 block 下采样，通道 128 -> 256，尺寸 28x28 -> 14x14。",
    nodes: [
      { type: "feature", layer: "layer3", title: "输入", dims: "[128,28,28]", size: "small", depth: 5 },
      { type: "kernel", title: "block1 conv1", kernel: "[256,128,3,3]", out: "stride=2" },
      { type: "kernel", title: "block1 conv2", kernel: "[256,256,3,3]", out: "stride=1" },
      { type: "kernel downsample", title: "downsample", kernel: "[256,128,1,1]", out: "stride=2" },
      { type: "add", title: "block1 残差相加", dims: "[256,14,14]" },
      { type: "kernel", title: "block2 conv1", kernel: "[256,256,3,3]", out: "stride=1" },
      { type: "kernel", title: "block2 conv2", kernel: "[256,256,3,3]", out: "stride=1" },
      { type: "add", title: "block2 残差相加", dims: "[256,14,14]" },
      { type: "feature", layer: "layer3", title: "layer3 输出", dims: "[256,14,14]", size: "tiny", depth: 6 },
    ],
  },
  layer4: {
    title: "layer4（2 个 BasicBlock）",
    subtitle: "第一个 block 下采样，通道 256 -> 512，尺寸 14x14 -> 7x7。",
    nodes: [
      { type: "feature", layer: "layer4", title: "输入", dims: "[256,14,14]", size: "tiny", depth: 6 },
      { type: "kernel", title: "block1 conv1", kernel: "[512,256,3,3]", out: "stride=2" },
      { type: "kernel", title: "block1 conv2", kernel: "[512,512,3,3]", out: "stride=1" },
      { type: "kernel downsample", title: "downsample", kernel: "[512,256,1,1]", out: "stride=2" },
      { type: "add", title: "block1 残差相加", dims: "[512,7,7]" },
      { type: "kernel", title: "block2 conv1", kernel: "[512,512,3,3]", out: "stride=1" },
      { type: "kernel", title: "block2 conv2", kernel: "[512,512,3,3]", out: "stride=1" },
      { type: "add", title: "block2 残差相加", dims: "[512,7,7]" },
      { type: "feature", layer: "layer4", title: "layer4 输出", dims: "[512,7,7]", size: "micro", depth: 7 },
    ],
  },
  head: {
    title: "分类头（项目中不作为主要展示）",
    subtitle: "原始 ResNet18 后面还有 avgpool、flatten 和 fc；你的项目主要展示前面的卷积特征图。",
    nodes: [
      { type: "feature", layer: "layer4", title: "输入", dims: "[512,7,7]", size: "micro", depth: 7 },
      { type: "pool", title: "avgpool", dims: "[512,1,1]" },
      { type: "fc", title: "flatten", dims: "[512]" },
      { type: "fc", title: "fc", dims: "[1000]" },
    ],
  },
};

const VGG_STAGE_TABS = [
  { id: "vgg_block1", label: "block1", layer: "conv1_2" },
  { id: "vgg_block2", label: "block2", layer: "conv2_2" },
  { id: "vgg_block3", label: "block3", layer: "conv3_3" },
  { id: "vgg_block4", label: "block4", layer: "conv4_3" },
  { id: "vgg_block5", label: "block5", layer: "conv5_3" },
  { id: "vgg_head", label: "分类头", layer: "conv5_3" },
];

const VGG_BLOCKS = [
  {
    id: "vgg_block1",
    title: "block1（2 个卷积层）",
    input: "[3, 224, 224]",
    output: "[64, 224, 224]",
    pool: "[64, 112, 112]",
    convs: [
      { name: "conv1_1", layer: "conv1_1", kernel: "[64, 3, 3, 3]", dims: "[64, 224, 224]" },
      { name: "conv1_2", layer: "conv1_2", kernel: "[64, 64, 3, 3]", dims: "[64, 224, 224]" },
    ],
  },
  {
    id: "vgg_block2",
    title: "block2（2 个卷积层）",
    input: "[64, 112, 112]",
    output: "[128, 112, 112]",
    pool: "[128, 56, 56]",
    convs: [
      { name: "conv2_1", layer: "conv2_1", kernel: "[128, 64, 3, 3]", dims: "[128, 112, 112]" },
      { name: "conv2_2", layer: "conv2_2", kernel: "[128, 128, 3, 3]", dims: "[128, 112, 112]" },
    ],
  },
  {
    id: "vgg_block3",
    title: "block3（3 个卷积层）",
    input: "[128, 56, 56]",
    output: "[256, 56, 56]",
    pool: "[256, 28, 28]",
    convs: [
      { name: "conv3_1", layer: "conv3_1", kernel: "[256, 128, 3, 3]", dims: "[256, 56, 56]" },
      { name: "conv3_2", layer: "conv3_2", kernel: "[256, 256, 3, 3]", dims: "[256, 56, 56]" },
      { name: "conv3_3", layer: "conv3_3", kernel: "[256, 256, 3, 3]", dims: "[256, 56, 56]" },
    ],
  },
  {
    id: "vgg_block4",
    title: "block4（3 个卷积层）",
    input: "[256, 28, 28]",
    output: "[512, 28, 28]",
    pool: "[512, 14, 14]",
    convs: [
      { name: "conv4_1", layer: "conv4_1", kernel: "[512, 256, 3, 3]", dims: "[512, 28, 28]" },
      { name: "conv4_2", layer: "conv4_2", kernel: "[512, 512, 3, 3]", dims: "[512, 28, 28]" },
      { name: "conv4_3", layer: "conv4_3", kernel: "[512, 512, 3, 3]", dims: "[512, 28, 28]" },
    ],
  },
  {
    id: "vgg_block5",
    title: "block5（3 个卷积层）",
    input: "[512, 14, 14]",
    output: "[512, 14, 14]",
    pool: "[512, 7, 7]",
    convs: [
      { name: "conv5_1", layer: "conv5_1", kernel: "[512, 512, 3, 3]", dims: "[512, 14, 14]" },
      { name: "conv5_2", layer: "conv5_2", kernel: "[512, 512, 3, 3]", dims: "[512, 14, 14]" },
      { name: "conv5_3", layer: "conv5_3", kernel: "[512, 512, 3, 3]", dims: "[512, 14, 14]" },
    ],
  },
];

const el = {
  modelStatus: document.getElementById("modelStatus"),
  navTabs: Array.from(document.querySelectorAll(".nav-tab")),
  views: Array.from(document.querySelectorAll(".view")),
  galleryGrid: document.getElementById("galleryGrid"),
  imageUpload: document.getElementById("imageUpload"),
  sourcePreview: document.getElementById("sourcePreview"),
  currentImageName: document.getElementById("currentImageName"),
  cnnImageSelect: document.getElementById("cnnImageSelect"),
  operationSelect: document.getElementById("operationSelect"),
  operationPrincipleTitle: document.getElementById("operationPrincipleTitle"),
  operationPrincipleText: document.getElementById("operationPrincipleText"),
  operationPrinciplePoints: document.getElementById("operationPrinciplePoints"),
  runClassical: document.getElementById("runClassical"),
  resizeMethodSelect: document.getElementById("resizeMethodSelect"),
  resizeScaleInput: document.getElementById("resizeScaleInput"),
  targetWidthInput: document.getElementById("targetWidthInput"),
  targetHeightInput: document.getElementById("targetHeightInput"),
  targetSizeInput: document.getElementById("targetSizeInput"),
  runResizeCompress: document.getElementById("runResizeCompress"),
  resizeInfoText: document.getElementById("resizeInfoText"),
  classicalResultTitle: document.getElementById("classicalResultTitle"),
  classicalResultMeta: document.getElementById("classicalResultMeta"),
  classicalSourceMeta: document.getElementById("classicalSourceMeta"),
  classicalResultSizeMeta: document.getElementById("classicalResultSizeMeta"),
  classicalSourceImage: document.getElementById("classicalSourceImage"),
  classicalResultImage: document.getElementById("classicalResultImage"),
  classicalLoading: document.getElementById("classicalLoading"),
  downloadClassicalLink: document.getElementById("downloadClassicalLink"),
  kernelPanel: document.getElementById("kernelPanel"),
  kernelGrid: document.getElementById("kernelGrid"),
  modelSelect: document.getElementById("modelSelect"),
  layerSelect: document.getElementById("layerSelect"),
  channelCount: document.getElementById("channelCount"),
  tileSize: document.getElementById("tileSize"),
  topChannelCount: document.getElementById("topChannelCount"),
  diagramStyle: document.getElementById("diagramStyle"),
  runCnn: document.getElementById("runCnn"),
  networkDiagram: document.getElementById("networkDiagram"),
  cnnResultTitle: document.getElementById("cnnResultTitle"),
  cnnResultMeta: document.getElementById("cnnResultMeta"),
  cnnResultImage: document.getElementById("cnnResultImage"),
  meanFeatureImage: document.getElementById("meanFeatureImage"),
  maxFeatureImage: document.getElementById("maxFeatureImage"),
  topChannelGrid: document.getElementById("topChannelGrid"),
  gradCamMeta: document.getElementById("gradCamMeta"),
  gradCamImage: document.getElementById("gradCamImage"),
  cnnLoading: document.getElementById("cnnLoading"),
  downloadCnnLink: document.getElementById("downloadCnnLink"),
};

async function init() {
  await loadExamples();
  await loadOptions();
  bindEvents();
  renderNetworkDiagram();
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  const data = await response.json();
  state.examples = data.examples || [];
  renderGallery();
  updateImageSelector();

  if (state.examples.length > 0) {
    selectExample(state.examples[0].name);
  }
}

async function loadOptions() {
  const response = await fetch("/api/options");
  const data = await response.json();

  state.cnnLayers = data.cnn_layers || {};
  state.classicalOperations = (data.classical_operations || []).map(mergeClassicalOperation);
  fillOperationSelect(el.operationSelect, state.classicalOperations);
  fillSelect(el.modelSelect, data.cnn_models || []);
  if (Array.from(el.modelSelect.options).some((option) => option.value === "resnet18")) {
    el.modelSelect.value = "resnet18";
    state.diagramStage = "layer1";
  }
  updateLayerOptions();
  updateModelStatus(data.model_status || {});
  updateClassicalPrinciple();
}

function bindEvents() {
  el.navTabs.forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  if (el.imageUpload) el.imageUpload.addEventListener("change", async () => {
    const file = el.imageUpload.files[0];
    if (!file) return;
    await saveUploadedImageToGallery(file);
  });

  if (el.cnnImageSelect) {
    el.cnnImageSelect.addEventListener("change", () => selectExample(el.cnnImageSelect.value));
  }
  if (el.modelSelect) el.modelSelect.addEventListener("change", updateLayerOptions);
  if (el.layerSelect) el.layerSelect.addEventListener("change", () => {
    state.diagramStage = stageFromLayer(el.layerSelect.value, state.diagramStage);
    renderNetworkDiagram();
  });
  if (el.diagramStyle) {
    el.diagramStyle.addEventListener("change", renderNetworkDiagram);
    el.diagramStyle.addEventListener("input", renderNetworkDiagram);
  }
  if (el.runCnn) el.runCnn.addEventListener("click", runCnn);
  if (el.runClassical) el.runClassical.addEventListener("click", runClassical);
  if (el.runResizeCompress) el.runResizeCompress.addEventListener("click", runResizeCompress);
  if (el.operationSelect) el.operationSelect.addEventListener("change", updateClassicalPrinciple);
}

function switchView(viewId) {
  el.navTabs.forEach((button) => button.classList.toggle("active", button.dataset.view === viewId));
  el.views.forEach((view) => view.classList.toggle("active", view.id === viewId));
}

function renderGallery() {
  el.galleryGrid.innerHTML = "";
  state.examples.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "gallery-card";
    if (item.name === state.selectedExample) card.classList.add("active");
    card.addEventListener("click", () => selectExample(item.name));

    const img = document.createElement("img");
    img.src = item.url;
    img.alt = item.label;
    const name = createText("div", "gallery-name", item.label);

    card.appendChild(img);
    card.appendChild(name);
    el.galleryGrid.appendChild(card);
  });
}

function selectExample(exampleName) {
  const item = state.examples.find((example) => example.name === exampleName);
  if (!item) return;
  state.uploadedFile = null;
  state.selectedExample = item.name;
  state.selectedImageLabel = item.label;
  el.imageUpload.value = "";
  el.sourcePreview.src = item.url;
  syncClassicalSourceImage();
  resetCnnPreprocess(true);
  updateCurrentImageName();
  renderGallery();
  updateImageSelector();
  renderNetworkDiagram();
}

function updateCurrentImageName() {
  el.currentImageName.textContent = state.selectedImageLabel
    ? `当前：${state.selectedImageLabel}`
    : "未选择图片";
}

async function saveUploadedImageToGallery(file) {
  const previewUrl = URL.createObjectURL(file);
  state.uploadedFile = file;
  state.selectedExample = "";
  state.selectedImageLabel = `${file.name}（保存中）`;
  el.sourcePreview.src = previewUrl;
  syncClassicalSourceImage();
  resetCnnPreprocess(true);
  updateCurrentImageName();

  const form = new FormData();
  form.append("image", file);

  try {
    const response = await fetch("/api/upload-example", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "保存图片失败");

    state.uploadedFile = null;
    await loadExamples();
    selectExample(data.name);
    switchView("galleryView");
  } catch (error) {
    state.selectedImageLabel = `${file.name}（保存失败：${error.message}）`;
    updateCurrentImageName();
  }
}

function fillSelect(select, options) {
  select.innerHTML = "";
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.weight_size ? `${item.label} (${item.weight_size})` : item.label;
    select.appendChild(option);
  });
}

function fillOperationSelect(select, options, config = {}) {
  if (!select) return;
  select.innerHTML = "";
  if (config.includeOriginal) {
    const option = document.createElement("option");
    option.value = "__original__";
    option.textContent = "原图";
    select.appendChild(option);
  }
  const groups = new Map();

  options.map(mergeClassicalOperation).forEach((item) => {
    const category = item.category || "其他处理";
    if (!groups.has(category)) {
      const group = document.createElement("optgroup");
      group.label = category;
      groups.set(category, group);
      select.appendChild(group);
    }

    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    groups.get(category).appendChild(option);
  });
}

function mergeClassicalOperation(item) {
  const fallback = CLASSICAL_OPERATION_FALLBACKS[item.id] || {};
  return {
    ...fallback,
    ...item,
    category: item.category || fallback.category,
    principle: item.principle || fallback.principle,
    principle_points: item.principle_points || fallback.principle_points || [],
  };
}

function updateClassicalPrinciple() {
  if (!el.operationSelect || !el.operationPrincipleTitle || !el.operationPrincipleText) return;
  const rawCurrent = state.classicalOperations.find((item) => item.id === el.operationSelect.value);
  const current = rawCurrent ? mergeClassicalOperation(rawCurrent) : null;
  if (!current) {
    el.operationPrincipleTitle.textContent = "技术原理";
    el.operationPrincipleText.textContent = "选择处理方法后显示该技术的基本原理。";
    renderPrinciplePoints([]);
    return;
  }

  el.operationPrincipleTitle.textContent = `${current.category || "图像处理"} · ${current.label}`;
  el.operationPrincipleText.textContent = current.principle || "当前方法缺少原理数据，请检查 /api/options 是否返回 principle 字段。";
  renderPrinciplePoints(current.principle_points || []);
}

function renderPrinciplePoints(points) {
  if (!el.operationPrinciplePoints) return;
  el.operationPrinciplePoints.innerHTML = "";
  points.forEach((point) => {
    const item = document.createElement("li");
    item.textContent = point;
    el.operationPrinciplePoints.appendChild(item);
  });
}

function updateImageSelector() {
  if (!el.cnnImageSelect) return;
  el.cnnImageSelect.innerHTML = "";
  state.examples.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.label;
    el.cnnImageSelect.appendChild(option);
  });
  if (state.selectedExample) {
    el.cnnImageSelect.value = state.selectedExample;
  }
}

function updateLayerOptions() {
  const modelId = el.modelSelect.value || "vgg16";
  fillSelect(el.layerSelect, state.cnnLayers[modelId] || []);
  renderNetworkDiagram();
}

function updateModelStatus(status) {
  el.modelStatus.classList.remove("ready", "warn");
  if (status.ready) {
    el.modelStatus.textContent = "CNN 依赖已就绪";
    el.modelStatus.classList.add("ready");
  } else {
    el.modelStatus.textContent = "CNN 依赖未安装";
    el.modelStatus.classList.add("warn");
  }
}

function buildFormData(options = {}) {
  const form = new FormData();
  if (options.useCnnPreprocess && state.cnnPreprocessedFile) {
    form.append("image", state.cnnPreprocessedFile);
  } else if (state.uploadedFile) {
    form.append("image", state.uploadedFile);
  } else {
    form.append("example", state.selectedExample);
  }
  return form;
}

async function applyCnnPreprocess(operation) {
  if (!operation) return;
  if (operation === "__original__") {
    resetCnnPreprocess(false);
    return;
  }
  const selectedOperation = state.classicalOperations.find((item) => item.id === operation);
  const form = buildFormData();
  form.append("operation", operation);

  try {
    const response = await fetch("/api/classical", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "预处理失败");

    const blob = await dataUrlToBlob(data.result);
    const operationLabel = data.title || selectedOperation?.label || "预处理图";
    state.cnnPreprocessedFile = new File([blob], `cnn_input_${operation}.png`, { type: blob.type || "image/png" });
    state.cnnPreprocessedUrl = data.result;
    state.cnnPreprocessLabel = operationLabel;
    state.cnnPreprocessOperation = operation;
    renderNetworkDiagram();
  } catch (error) {
    window.alert(`预处理失败：${error.message}`);
  }
}

function resetCnnPreprocess(silent) {
  state.cnnPreprocessedFile = null;
  state.cnnPreprocessedUrl = "";
  state.cnnPreprocessLabel = "";
  state.cnnPreprocessOperation = "__original__";
  renderNetworkDiagram();
}

function getCnnInputImageSrc() {
  return state.cnnPreprocessedUrl || el.sourcePreview?.src || "";
}

async function dataUrlToBlob(dataUrl) {
  const response = await fetch(dataUrl);
  return response.blob();
}

async function runClassical() {
  const form = buildFormData();
  form.append("operation", el.operationSelect.value);
  syncClassicalSourceImage();
  setLoading(el.classicalLoading, el.runClassical, true);

  try {
    const response = await fetch("/api/classical", { method: "POST", body: form });
    const data = await readJsonResponse(response, "图像处理");
    if (!response.ok) throw new Error(data.error || "处理失败");

    el.classicalResultImage.src = data.result;
    el.downloadClassicalLink.href = data.result;
    el.downloadClassicalLink.classList.remove("hidden");
    el.classicalResultTitle.textContent = data.title;
    el.classicalResultMeta.textContent = data.description;
    updateClassicalImageMeta(data);
    renderKernel(data.kernel);
  } catch (error) {
    el.classicalResultTitle.textContent = "处理失败";
    el.classicalResultMeta.textContent = error.message;
    el.classicalResultImage.removeAttribute("src");
    el.downloadClassicalLink.classList.add("hidden");
    el.kernelPanel.classList.add("hidden");
  } finally {
    setLoading(el.classicalLoading, el.runClassical, false);
  }
}

async function runResizeCompress() {
  const form = buildFormData();
  form.append("method", el.resizeMethodSelect ? el.resizeMethodSelect.value : "linear");
  form.append("scale", getResizeScale());
  form.append("target_width", el.targetWidthInput ? el.targetWidthInput.value : "");
  form.append("target_height", el.targetHeightInput ? el.targetHeightInput.value : "");
  form.append("target_kb", el.targetSizeInput ? el.targetSizeInput.value : "");
  syncClassicalSourceImage();
  setLoading(el.classicalLoading, el.runResizeCompress, true);

  try {
    const response = await fetch("/api/resize-compress", { method: "POST", body: form });
    const data = await readJsonResponse(response, "放缩 / 压缩");
    if (!response.ok) throw new Error(data.error || "放缩 / 压缩失败");

    el.classicalResultImage.src = data.result;
    el.downloadClassicalLink.href = data.result;
    el.downloadClassicalLink.classList.remove("hidden");
    el.classicalResultTitle.textContent = data.title;
    const info = formatResizeInfo(data);
    el.classicalResultMeta.textContent = `${data.description} ${info}`;
    if (el.resizeInfoText) el.resizeInfoText.textContent = info;
    updateClassicalImageMeta(data);
    el.kernelPanel.classList.add("hidden");
  } catch (error) {
    el.classicalResultTitle.textContent = "放缩 / 压缩失败";
    el.classicalResultMeta.textContent = error.message;
    if (el.resizeInfoText) el.resizeInfoText.textContent = error.message;
    el.classicalResultImage.removeAttribute("src");
    el.downloadClassicalLink.classList.add("hidden");
    el.kernelPanel.classList.add("hidden");
  } finally {
    setLoading(el.classicalLoading, el.runResizeCompress, false);
  }
}

function formatResizeInfo(data) {
  const originalSize = `${data.original_width} x ${data.original_height}`;
  const resultSize = `${data.result_width} x ${data.result_height}`;
  const originalKb = formatKb(data.original_kb);
  const resultKb = formatKb(data.result_kb);
  const quality = data.quality ? `，JPEG 质量 ${data.quality}` : "";
  return `输入尺寸 ${originalSize}，输出尺寸 ${resultSize}；输入约 ${originalKb}，输出约 ${resultKb}${quality}。`;
}

function updateClassicalImageMeta(data) {
  if (el.classicalSourceMeta) {
    el.classicalSourceMeta.textContent = `尺寸 ${data.original_width} x ${data.original_height}，大小约 ${formatKb(data.original_kb)}`;
  }
  if (el.classicalResultSizeMeta) {
    el.classicalResultSizeMeta.textContent = `尺寸 ${data.result_width} x ${data.result_height}，大小约 ${formatKb(data.result_kb)}`;
  }
}

function getResizeScale() {
  const percent = Number(el.resizeScaleInput ? el.resizeScaleInput.value : 100);
  if (!Number.isFinite(percent) || percent <= 0) return "1";
  return String(Math.max(10, Math.min(percent, 400)) / 100);
}

async function readJsonResponse(response, actionName) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    const isHtml = text.trim().startsWith("<");
    const hint = isHtml
      ? `${actionName}接口返回了 HTML 页面，不是 JSON。通常是 Flask 没重启、路由没生效，或者请求地址返回了 404。`
      : `${actionName}接口返回内容不是合法 JSON。`;
    throw new Error(hint);
  }
}

function formatKb(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "未知";
  if (number >= 1024) return `${(number / 1024).toFixed(2)} MB`;
  return `${number.toFixed(1)} KB`;
}

async function runCnn() {
  const form = buildFormData({ useCnnPreprocess: true });
  form.append("model", el.modelSelect.value);
  form.append("layer", el.layerSelect.value);
  form.append("channels", el.channelCount.value);
  form.append("tile_size", el.tileSize.value);
  form.append("top_k", el.topChannelCount ? el.topChannelCount.value : "6");
  setLoading(el.cnnLoading, el.runCnn, true);

  try {
    const response = await fetch("/api/cnn", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "处理失败");

    el.cnnResultImage.src = data.result;
    el.cnnResultImage.classList.add("feature-grid");
    setOptionalImage(el.meanFeatureImage, data.mean_result);
    setOptionalImage(el.maxFeatureImage, data.max_result);
    renderTopChannels(data.top_channels || []);
    el.downloadCnnLink.href = data.result;
    el.downloadCnnLink.classList.remove("hidden");
    el.cnnResultTitle.textContent = `${data.model} - ${data.layer}`;
    const summary = data.summary || {};
    const maxChannelIndex = Number(summary.max_channel_index);
    const maxChannelScore = Number(summary.max_channel_score);
    const maxChannelText = Number.isInteger(maxChannelIndex)
      ? `，最大响应通道：ch ${maxChannelIndex + 1}${Number.isFinite(maxChannelScore) ? `（平均响应 ${maxChannelScore.toFixed(3)}）` : ""}`
      : "";
    const totalChannels = Number(data.total_channels);
    const channelText = Number.isInteger(totalChannels)
      ? `显示 ${data.channels} / 共 ${totalChannels} 个通道`
      : `显示 ${data.channels} 个通道`;
    const inputText = state.cnnPreprocessLabel ? `，CNN 输入：${state.cnnPreprocessLabel}` : "，CNN 输入：原图";
    el.cnnResultMeta.textContent = `输出张量形状：${data.shape.join(" x ")}，${channelText}${maxChannelText}${inputText}`;
    await runGradCam();
  } catch (error) {
    el.cnnResultTitle.textContent = "处理失败";
    el.cnnResultMeta.textContent = error.message;
    el.cnnResultImage.removeAttribute("src");
    clearOptionalImage(el.meanFeatureImage);
    clearOptionalImage(el.maxFeatureImage);
    renderTopChannels([]);
    el.gradCamMeta.textContent = "Grad-CAM 未生成。";
    el.gradCamImage.removeAttribute("src");
    el.downloadCnnLink.classList.add("hidden");
  } finally {
    setLoading(el.cnnLoading, el.runCnn, false);
  }
}

async function runGradCam() {
  const form = buildFormData({ useCnnPreprocess: true });
  form.append("model", el.modelSelect.value);
  form.append("layer", el.layerSelect.value);
  el.gradCamMeta.textContent = "Grad-CAM 生成中...";
  el.gradCamImage.removeAttribute("src");

  try {
    const response = await fetch("/api/gradcam", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Grad-CAM 生成失败");

    el.gradCamImage.src = data.result;
    const classLabel = formatClassLabel(data.class_label, data.class_index);
    const score = Number(data.score);
    el.gradCamMeta.textContent = `反传层：${data.layer}，预测类别索引：${data.class_index}，英文标签：${classLabel}，分数：${Number.isFinite(score) ? score.toFixed(3) : "未知"}`;
  } catch (error) {
    el.gradCamMeta.textContent = error.message;
    el.gradCamImage.removeAttribute("src");
  }
}

function formatClassLabel(label, classIndex) {
  const text = typeof label === "string" ? label.trim() : "";
  if (text) return text;
  const indexText = classIndex === undefined || classIndex === null ? "类别索引未知" : `ImageNet class ${classIndex}`;
  return `${indexText}（当前 torchvision 未提供英文类别名）`;
}

function renderNetworkDiagram() {
  if (!el.networkDiagram || !el.modelSelect || !el.layerSelect) return;

  const modelId = el.modelSelect.value || "vgg16";
  const selectedLayer = el.layerSelect.value;
  el.networkDiagram.innerHTML = "";

  const style = el.diagramStyle ? el.diagramStyle.value : "scene";
  if (modelId === "resnet18" && style === "blueprint") {
    el.networkDiagram.appendChild(renderResNetBlueprint(selectedLayer));
    return;
  }

  if (modelId === "resnet18") {
    el.networkDiagram.appendChild(renderResNetStageDiagram(selectedLayer));
    return;
  }

  if (modelId === "vgg16" && style === "blueprint") {
    el.networkDiagram.appendChild(renderVggBlueprint(selectedLayer));
    return;
  }

  if (modelId === "vgg16") {
    el.networkDiagram.appendChild(renderVggStageDiagram(selectedLayer));
    return;
  }

  const scene = MODEL_SCENES[modelId] || [];

  const sceneEl = document.createElement("div");
  sceneEl.className = `layer-scene ${modelId}`;

  scene.forEach((node, index) => {
    sceneEl.appendChild(renderSceneNode(node, selectedLayer));
    if (index < scene.length - 1) {
      sceneEl.appendChild(createText("div", "scene-connector", ""));
    }
  });

  el.networkDiagram.appendChild(sceneEl);
  el.networkDiagram.appendChild(createText("div", "scene-hint", "说明：图中每个蓝色立体块表示一个特征图张量；小立方体表示卷积核组；粉色薄块表示池化；圆形加号表示残差相加。"));
}

function renderVggStageDiagram(selectedLayer) {
  const stageId = VGG_STAGE_TABS.some((item) => item.id === state.diagramStage)
    ? state.diagramStage
    : vggStageFromLayer(selectedLayer, "vgg_block1");
  const wrap = document.createElement("div");
  wrap.className = "stage-diagram vgg-stage-diagram";

  const tabs = document.createElement("div");
  tabs.className = "stage-tabs";
  VGG_STAGE_TABS.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "stage-tab";
    if (item.id === stageId) button.classList.add("active");
    button.textContent = item.label;
    button.addEventListener("click", () => {
      state.diagramStage = item.id;
      renderNetworkDiagram();
    });
    tabs.appendChild(button);
  });
  wrap.appendChild(tabs);

  const focus = document.createElement("div");
  focus.className = `vgg-focus ${stageId}`;
  focus.appendChild(renderInputPanel());
  focus.appendChild(renderPanelArrow());

  if (stageId === "vgg_head") {
    focus.appendChild(renderVggHeadPanel());
  } else {
    focus.appendChild(renderVggBlockPanel(getVggBlock(stageId), selectedLayer));
  }

  wrap.appendChild(focus);
  wrap.appendChild(renderVggLegend());
  wrap.appendChild(createText("div", "scene-hint", "点击上方 block 按钮切换展示的 VGG16 阶段；点击蓝色特征图堆可以生成对应卷积层的通道特征图。"));
  return wrap;
}

function renderVggBlueprint(selectedLayer) {
  const wrap = document.createElement("div");
  wrap.className = "vgg-blueprint";

  const flow = document.createElement("div");
  flow.className = "vgg-flow";
  flow.appendChild(renderInputPanel());
  VGG_BLOCKS.forEach((block) => {
    flow.appendChild(renderPanelArrow());
    flow.appendChild(renderVggBlockPanel(block, selectedLayer));
  });
  flow.appendChild(renderPanelArrow());
  flow.appendChild(renderVggHeadPanel());

  wrap.appendChild(flow);
  wrap.appendChild(renderVggLegend());
  wrap.appendChild(createText("div", "resnet-title", "VGG16 结构流程图"));
  return wrap;
}

function getVggBlock(stageId) {
  return VGG_BLOCKS.find((block) => block.id === stageId) || VGG_BLOCKS[0];
}

function renderVggBlockPanel(block, selectedLayer) {
  const panel = createText("section", "bp-panel vgg-block-panel", "");
  panel.appendChild(createText("div", "bp-panel-title", block.title));
  panel.appendChild(createText("div", "vgg-io-label", `输入 ${block.input}`));

  const body = createText("div", "vgg-block-body", "");
  block.convs.forEach((conv, index) => {
    body.appendChild(renderVggConvStep(conv, selectedLayer));
    if (index < block.convs.length - 1) body.appendChild(renderDownArrow());
  });
  panel.appendChild(body);
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "vgg-pool-label", `maxpool 2 x 2\nstride=2\n输出 ${block.pool}`));
  panel.appendChild(createText("div", "bp-final", `${block.id.replace("vgg_", "")} 输出 ${block.output}`));
  return panel;
}

function renderVggConvStep(conv, selectedLayer) {
  const step = createText("div", "vgg-conv-step", "");
  step.appendChild(renderMiniStack(`feature ${vggFeatureSizeClass(conv.dims)}`, conv.layer, selectedLayer, conv.dims));
  step.appendChild(createText("div", "vgg-conv-text", `${conv.name}\n卷积核 ${conv.kernel}\npadding=1, ReLU`));
  return step;
}

function renderVggHeadPanel() {
  const panel = createText("section", "bp-panel vgg-head-panel", "");
  panel.appendChild(createText("div", "bp-panel-title", "分类头"));
  panel.appendChild(createText("div", "bp-op-text", "输入来自 block5\n[512, 7, 7]"));
  panel.appendChild(renderMiniStack("feature tiny", "conv5_3", el.layerSelect.value, "[512, 7, 7]"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "flatten\n512 x 7 x 7 = 25088"));
  panel.appendChild(createText("div", "bp-vector wide", "25088"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "fc1 / fc2\n4096 -> 4096"));
  panel.appendChild(createText("div", "bp-vector", "4096"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "fc3\n1000 类"));
  panel.appendChild(createText("div", "bp-vector", "1000"));
  panel.appendChild(createText("div", "bp-dims strong", "输出：[1000]"));
  return panel;
}

function renderVggLegend() {
  const legend = createText("div", "bp-legend vgg-legend", "");
  legend.appendChild(renderLegendItem("feature", "特征图\n多通道特征张量 [C, H, W]"));
  legend.appendChild(renderLegendItem("kernel", "3 x 3 卷积\nstride=1, padding=1"));
  legend.appendChild(renderLegendItem("arrow", "顺序前向传播"));
  legend.appendChild(renderLegendItem("down", "maxpool 2 x 2\n空间尺寸减半"));
  return legend;
}

function vggStageFromLayer(layerId, fallback) {
  if (String(layerId).startsWith("conv1_")) return "vgg_block1";
  if (String(layerId).startsWith("conv2_")) return "vgg_block2";
  if (String(layerId).startsWith("conv3_")) return "vgg_block3";
  if (String(layerId).startsWith("conv4_")) return "vgg_block4";
  if (String(layerId).startsWith("conv5_")) return "vgg_block5";
  return fallback || "vgg_block1";
}

function vggFeatureSizeClass(dims) {
  const numbers = String(dims).match(/\d+/g) || [];
  const spatial = Number(numbers[numbers.length - 1]);
  if (spatial >= 224) return "huge";
  if (spatial >= 112) return "big";
  if (spatial >= 56) return "bigger";
  if (spatial >= 28) return "small";
  return "tiny";
}

function renderResNetStageDiagram(selectedLayer) {
  const stageId = RESNET_STAGE_SCENES[state.diagramStage] ? state.diagramStage : "stem";
  const wrap = document.createElement("div");
  wrap.className = "stage-diagram";

  const tabs = document.createElement("div");
  tabs.className = "stage-tabs";
  RESNET_STAGE_TABS.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "stage-tab";
    if (item.id === stageId) button.classList.add("active");
    button.textContent = item.label;
    button.addEventListener("click", () => {
      state.diagramStage = item.id;
      renderNetworkDiagram();
    });
    tabs.appendChild(button);
  });
  wrap.appendChild(tabs);

  const focus = document.createElement("div");
  focus.className = `resnet-focus ${stageId}`;

  focus.appendChild(renderInputPanel());
  focus.appendChild(renderPanelArrow());
  if (stageId === "stem") {
    focus.appendChild(renderStemPanel(selectedLayer));
  } else if (stageId === "head") {
    focus.appendChild(renderHeadPanel());
  } else {
    focus.appendChild(renderStagePanel(getStagePanelData(stageId), selectedLayer));
  }

  wrap.appendChild(focus);
  wrap.appendChild(renderLegend());
  wrap.appendChild(createText("div", "scene-hint", "点击上方阶段按钮切换展示的 layer；点击图中的特征图堆可以生成对应层的通道特征图。"));
  return wrap;
}

function getStagePanelData(stageId) {
  const stages = {
    layer1: {
      id: "layer1",
      title: "layer1（2 个 BasicBlock）",
      final: "[64, 56, 56]",
      blocks: [
        { name: "block1", input: "[64, 56, 56]", inputLayer: "conv1", conv1: "[64, 64, 3, 3]", conv1Layer: "layer1_block1_conv1", conv2: "[64, 64, 3, 3]", conv2Layer: "layer1_block1_conv2", output: "[64, 56, 56]", outputLayer: "layer1_block1_out" },
        { name: "block2", input: "[64, 56, 56]", inputLayer: "layer1_block1_out", conv1: "[64, 64, 3, 3]", conv1Layer: "layer1_block2_conv1", conv2: "[64, 64, 3, 3]", conv2Layer: "layer1_block2_conv2", output: "[64, 56, 56]", outputLayer: "layer1_block2_out" },
      ],
    },
    layer2: {
      id: "layer2",
      title: "layer2（2 个 BasicBlock）",
      final: "[128, 28, 28]",
      blocks: [
        { name: "block1", input: "[64, 56, 56]", inputLayer: "layer1", conv1: "[128, 64, 3, 3]", conv1Layer: "layer2_block1_conv1", conv2: "[128, 128, 3, 3]", conv2Layer: "layer2_block1_conv2", downsample: "[128, 64, 1, 1]", downsampleLayer: "layer2_block1_downsample", output: "[128, 28, 28]", outputLayer: "layer2_block1_out" },
        { name: "block2", input: "[128, 28, 28]", inputLayer: "layer2_block1_out", conv1: "[128, 128, 3, 3]", conv1Layer: "layer2_block2_conv1", conv2: "[128, 128, 3, 3]", conv2Layer: "layer2_block2_conv2", output: "[128, 28, 28]", outputLayer: "layer2_block2_out" },
      ],
    },
    layer3: {
      id: "layer3",
      title: "layer3（2 个 BasicBlock）",
      final: "[256, 14, 14]",
      blocks: [
        { name: "block1", input: "[128, 28, 28]", inputLayer: "layer2", conv1: "[256, 128, 3, 3]", conv1Layer: "layer3_block1_conv1", conv2: "[256, 256, 3, 3]", conv2Layer: "layer3_block1_conv2", downsample: "[256, 128, 1, 1]", downsampleLayer: "layer3_block1_downsample", output: "[256, 14, 14]", outputLayer: "layer3_block1_out" },
        { name: "block2", input: "[256, 14, 14]", inputLayer: "layer3_block1_out", conv1: "[256, 256, 3, 3]", conv1Layer: "layer3_block2_conv1", conv2: "[256, 256, 3, 3]", conv2Layer: "layer3_block2_conv2", output: "[256, 14, 14]", outputLayer: "layer3_block2_out" },
      ],
    },
    layer4: {
      id: "layer4",
      title: "layer4（2 个 BasicBlock）",
      final: "[512, 7, 7]",
      blocks: [
        { name: "block1", input: "[256, 14, 14]", inputLayer: "layer3", conv1: "[512, 256, 3, 3]", conv1Layer: "layer4_block1_conv1", conv2: "[512, 512, 3, 3]", conv2Layer: "layer4_block1_conv2", downsample: "[512, 256, 1, 1]", downsampleLayer: "layer4_block1_downsample", output: "[512, 7, 7]", outputLayer: "layer4_block1_out" },
        { name: "block2", input: "[512, 7, 7]", inputLayer: "layer4_block1_out", conv1: "[512, 512, 3, 3]", conv1Layer: "layer4_block2_conv1", conv2: "[512, 512, 3, 3]", conv2Layer: "layer4_block2_conv2", output: "[512, 7, 7]", outputLayer: "layer4_block2_out" },
      ],
    },
  };
  return stages[stageId] || stages.layer1;
}

function stageFromLayer(layerId, fallback) {
  if (layerId === "conv1") return "stem";
  if (["layer1", "layer2", "layer3", "layer4"].includes(layerId)) return layerId;
  if (String(layerId).startsWith("conv")) return vggStageFromLayer(layerId, fallback);
  return fallback || "stem";
}

function renderResNetBlueprint(selectedLayer) {
  const wrap = document.createElement("div");
  wrap.className = "resnet-blueprint";

  const flow = document.createElement("div");
  flow.className = "resnet-flow";
  flow.appendChild(renderInputPanel());
  flow.appendChild(renderPanelArrow());
  flow.appendChild(renderStemPanel(selectedLayer));

  const stages = ["layer1", "layer2", "layer3", "layer4"].map(getStagePanelData);

  stages.forEach((stage) => {
    flow.appendChild(renderPanelArrow());
    flow.appendChild(renderStagePanel(stage, selectedLayer));
  });

  flow.appendChild(renderPanelArrow());
  flow.appendChild(renderHeadPanel());
  wrap.appendChild(flow);
  wrap.appendChild(renderLegend());
  wrap.appendChild(createText("div", "resnet-title", "ResNet18 结构流程图"));
  return wrap;
}

function renderInputPanel() {
  const panel = createText("section", "bp-panel input-panel input-switch-panel", "");
  const control = createText("div", "bp-input-switcher", "");
  control.appendChild(createText("label", "bp-input-switcher-label", "选择输入图"));
  control.appendChild(renderInputDropdown());

  const img = document.createElement("img");
  img.src = getCnnInputImageSrc();
  img.alt = "输入图像";
  img.className = "bp-input-img";
  panel.appendChild(img);
  panel.appendChild(control);
  panel.appendChild(createText("div", "bp-input-title", state.cnnPreprocessLabel ? "CNN 输入图像" : "输入图像"));
  if (state.cnnPreprocessLabel) {
    panel.appendChild(createText("div", "bp-input-method", state.cnnPreprocessLabel));
  }
  panel.appendChild(createText("div", "bp-dims strong", "[3, 224, 224]"));
  return panel;
}

function renderInputDropdown() {
  const dropdown = createText("div", "bp-input-dropdown", "");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "bp-input-select-button";
  button.textContent = getInputOperationLabel();

  const menu = createText("div", "bp-input-menu hidden", "");
  appendInputMenuItem(menu, "__original__", "原图", "原图");

  const groups = new Map();
  state.classicalOperations.map(mergeClassicalOperation).forEach((operation) => {
    const category = operation.category || "其他处理";
    if (!groups.has(category)) {
      const group = createText("div", "bp-input-menu-group", "");
      group.appendChild(createText("div", "bp-input-menu-title", category));
      groups.set(category, group);
      menu.appendChild(group);
    }
    appendInputMenuItem(groups.get(category), operation.id, operation.label, category);
  });

  button.addEventListener("click", (event) => {
    event.stopPropagation();
    menu.classList.toggle("hidden");
  });

  dropdown.appendChild(button);
  dropdown.appendChild(menu);
  return dropdown;
}

function appendInputMenuItem(parent, value, label, category) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = "bp-input-menu-item";
  if (value === state.cnnPreprocessOperation) item.classList.add("active");
  item.textContent = label;
  item.addEventListener("click", async (event) => {
    event.stopPropagation();
    item.disabled = true;
    await applyCnnPreprocess(value);
    item.disabled = false;
  });
  parent.appendChild(item);
}

function getInputOperationLabel() {
  if (!state.cnnPreprocessOperation || state.cnnPreprocessOperation === "__original__") return "原图";
  const operation = state.classicalOperations.find((item) => item.id === state.cnnPreprocessOperation);
  return operation?.label || state.cnnPreprocessLabel || "已处理输入";
}

function renderStemPanel(selectedLayer) {
  const panel = createText("section", "bp-panel stem-panel", "");
  panel.appendChild(createText("div", "bp-panel-title", "Stem（输入层）"));
  panel.appendChild(createText("div", "bp-op-text", "conv1: 卷积核 [64, 3, 7, 7]\nstride=2, padding=3"));
  panel.appendChild(renderMiniStack("feature big", "conv1", selectedLayer, "[64, 112, 112]"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "bn1 + relu: 尺寸不变"));
  panel.appendChild(renderMiniStack("feature big", "conv1", selectedLayer, "[64, 112, 112]"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "maxpool: 3 x 3\nstride=2, padding=1"));
  panel.appendChild(renderMiniStack("feature bigger", "layer1", selectedLayer, "[64, 56, 56]"));
  return panel;
}

function renderStagePanel(stage, selectedLayer) {
  const panel = createText("section", "bp-panel stage-panel", "");
  panel.appendChild(createText("div", "bp-panel-title", stage.title));

  const blocks = document.createElement("div");
  blocks.className = "bp-blocks";
  stage.blocks.forEach((block) => blocks.appendChild(renderNeatBasicBlock(stage.id, block, selectedLayer)));
  panel.appendChild(blocks);
  panel.appendChild(createText("div", "bp-final", `${stage.id} 最终输出 ${stage.final}`));
  return panel;
}

function renderBasicBlock(layerId, block, selectedLayer) {
  const box = createText("div", "bp-basic-block", "");
  box.appendChild(createText("div", "bp-block-title", block.name));
  box.appendChild(createText("div", "bp-small-label", `输入\n${block.input}`));
  box.appendChild(renderMiniStack("kernel", layerId, selectedLayer, ""));
  box.appendChild(renderDownArrow());
  box.appendChild(createText("div", "bp-conv-label", `conv1\n${block.conv1}\nstride=${block.downsample ? "2" : "1"}`));
  box.appendChild(renderMiniStack("kernel small", layerId, selectedLayer, ""));
  box.appendChild(renderDownArrow());
  box.appendChild(createText("div", "bp-conv-label", `conv2\n${block.conv2}\nstride=1`));

  const join = document.createElement("div");
  join.className = "bp-join-row";
  join.appendChild(renderAddNode());
  if (block.downsample) {
    const down = createText("div", "bp-downsample", `downsample\n1 x 1 conv\n${block.downsample}\nstride=2`);
    down.appendChild(renderMiniStack("downsample-stack", layerId, selectedLayer, ""));
    join.appendChild(down);
  }
  box.appendChild(join);
  box.appendChild(createText("div", "bp-small-label", `输出\n${block.output}`));
  return box;
}

function renderLargeBasicBlock(layerId, block, selectedLayer) {
  const box = createText("div", "bp-basic-block large-block", "");
  box.classList.add(block.downsample ? "projection-skip" : "identity-skip");
  box.appendChild(createText("div", "bp-block-title", block.name));
  box.appendChild(createText("div", "bp-small-label", `输入\n${block.input}`));
  box.appendChild(renderMiniStack("feature block-map", layerId, selectedLayer, ""));
  box.appendChild(renderDownArrow());
  box.appendChild(renderConvInfo("conv1", block.conv1, block.downsample ? "2" : "1"));
  box.appendChild(renderMiniStack("feature block-map", layerId, selectedLayer, ""));
  box.appendChild(renderDownArrow());
  box.appendChild(renderConvInfo("conv2", block.conv2, "1"));

  const join = document.createElement("div");
  join.className = "bp-join-row";
  join.appendChild(renderAddNode());
  if (block.downsample) {
    const down = createText("div", "bp-downsample", `downsample\n1 x 1 conv\n${block.downsample}\nstride=2`);
    down.appendChild(renderMiniStack("feature downsample-stack", layerId, selectedLayer, ""));
    join.appendChild(down);
  }
  box.appendChild(join);
  box.appendChild(createText("div", "bp-small-label", `输出\n${block.output}`));
  return box;
}

function renderConvInfo(name, kernel, stride) {
  const info = createText("div", "bp-conv-info", "");
  info.appendChild(renderKernelIcon(kernel.includes("1, 1") ? "k1" : "k3"));
  info.appendChild(createText("div", "bp-conv-label", `${name}\n${kernel}\nstride=${stride}`));
  return info;
}

function renderKernelIcon(type) {
  const icon = document.createElement("span");
  icon.className = `bp-kernel-icon ${type}`;
  const cells = type === "k1" ? 1 : 9;
  for (let i = 0; i < cells; i += 1) {
    icon.appendChild(document.createElement("i"));
  }
  return icon;
}

function renderPptBasicBlock(layerId, block, selectedLayer) {
  const box = createText("div", "ppt-basic-block", "");
  box.classList.add(block.downsample ? "projection-skip" : "identity-skip");
  box.appendChild(createText("div", "bp-block-title", block.name));

  const flow = createText("div", "ppt-flow", "");
  flow.appendChild(createText("div", "ppt-io-label", `输入\n${block.input}`));
  flow.appendChild(renderPptCube("feature input", layerId, selectedLayer));
  flow.appendChild(renderPptArrow("down"));
  flow.appendChild(renderPptConv("conv1", block.conv1, block.downsample ? "2" : "1"));
  flow.appendChild(renderPptCube("feature mid", layerId, selectedLayer));
  flow.appendChild(renderPptArrow("down"));
  flow.appendChild(renderPptConv("conv2", block.conv2, "1"));
  flow.appendChild(renderPptCube("feature output", layerId, selectedLayer));

  const addRow = createText("div", "ppt-add-row", "");
  addRow.appendChild(renderAddNode());
  addRow.appendChild(createText("span", "ppt-add-label", "相加"));
  flow.appendChild(addRow);
  flow.appendChild(createText("div", "ppt-io-label output", `输出\n${block.output}`));
  box.appendChild(flow);

  const skip = createText("div", "ppt-skip", "");
  if (block.downsample) {
    const down = createText("div", "ppt-downsample", `Downsample卷积:\n${block.downsample}\nstride=2`);
    down.appendChild(renderPptKernel("k1"));
    skip.appendChild(down);
  }
  box.appendChild(skip);
  return box;
}

function renderPptConv(name, kernel, stride) {
  const row = createText("div", "ppt-conv-row", "");
  row.appendChild(createText("div", "ppt-conv-text", `${name}:\n卷积核 ${kernel}\nstride=${stride}`));
  row.appendChild(renderPptKernel(kernel.includes("1, 1") ? "k1" : "k3"));
  return row;
}

function renderPptCube(kind, layerId, selectedLayer) {
  const cube = document.createElement(layerId ? "button" : "div");
  if (layerId) cube.type = "button";
  cube.className = `ppt-cube ${kind}`;
  if (layerId) {
    cube.classList.add("clickable");
    cube.addEventListener("click", () => selectLayerAndRun(layerId));
  }
  if (layerId && layerId === selectedLayer) cube.classList.add("active");
  cube.appendChild(createText("span", "ppt-cube-face", ""));
  return cube;
}

function renderPptKernel(kind) {
  const kernel = createText("span", `ppt-kernel ${kind}`, "");
  const cells = kind === "k1" ? 1 : 9;
  for (let i = 0; i < cells; i += 1) {
    kernel.appendChild(document.createElement("i"));
  }
  return kernel;
}

function renderPptArrow(direction) {
  return createText("div", `ppt-arrow ${direction}`, "");
}

function renderNeatBasicBlock(layerId, block, selectedLayer) {
  const box = createText("div", "neat-block", "");
  box.classList.add(block.downsample ? "projection-skip" : "identity-skip");
  box.appendChild(createText("div", "neat-block-title", block.name));
  box.appendChild(createText("div", "neat-input", `输入\n${block.input}`));

  const main = createText("div", "neat-main", "");
  main.appendChild(renderNeatStack(block.inputLayer || layerId, selectedLayer, tensorSizeClass(block.input)));
  main.appendChild(renderNeatArrow());
  main.appendChild(renderNeatConv("conv1", block.conv1, block.downsample ? "2" : "1", block.conv1Layer, block.output));
  main.appendChild(renderNeatArrow());
  main.appendChild(renderNeatConv("conv2", block.conv2, "1", block.conv2Layer, block.output));
  main.appendChild(renderNeatArrow());
  main.appendChild(renderClickableAdd(block.outputLayer, selectedLayer));
  main.appendChild(createText("div", "neat-output", `输出\n${block.output}`));
  box.appendChild(main);

  const skip = createText("div", "neat-skip", "");
  if (block.downsample) {
    const down = createText("div", "neat-downsample", `downsample\n1 x 1 conv\n${block.downsample}\nstride=2`);
    down.appendChild(renderNeatStack(block.downsampleLayer, selectedLayer, `downsample ${tensorSizeClass(block.output)}`));
    skip.appendChild(down);
  }
  box.appendChild(skip);
  return box;
}

function renderNeatConv(name, kernel, stride, layerId = "", featureDims = "") {
  const row = createText("div", "neat-conv", "");
  row.appendChild(renderNeatStack(layerId, el.layerSelect.value, `conv-feature ${tensorSizeClass(featureDims)}`));
  row.appendChild(createText("div", "neat-conv-text", `${name}\n卷积核 ${kernel}\nstride=${stride}`));
  return row;
}

function renderNeatKernel(layerId, selectedLayer, kernel) {
  const cube = document.createElement(layerId ? "button" : "div");
  if (layerId) cube.type = "button";
  cube.className = `neat-kernel-cube ${kernel.includes("1, 1") ? "k1" : "k3"}`;
  if (layerId) {
    cube.classList.add("clickable");
    cube.addEventListener("click", () => selectLayerAndRun(layerId));
  }
  if (layerId && layerId === selectedLayer) cube.classList.add("active");

  const front = createText("span", "kernel-front", "");
  const cellCount = kernel.includes("1, 1") ? 1 : 9;
  for (let i = 0; i < cellCount; i += 1) {
    front.appendChild(document.createElement("i"));
  }
  cube.appendChild(front);
  return cube;
}

function renderClickableAdd(layerId, selectedLayer) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "neat-add-button";
  if (layerId) {
    button.classList.add("clickable");
    button.addEventListener("click", () => selectLayerAndRun(layerId));
  }
  if (layerId && layerId === selectedLayer) button.classList.add("active");
  button.appendChild(renderAddNode());
  return button;
}

function renderNeatStack(layerId, selectedLayer, extra = "") {
  const stack = document.createElement(layerId ? "button" : "div");
  if (layerId) stack.type = "button";
  stack.className = `neat-stack ${extra}`;
  if (layerId) {
    stack.classList.add("clickable");
    stack.addEventListener("click", () => selectLayerAndRun(layerId));
  }
  if (layerId && layerId === selectedLayer) stack.classList.add("active");
  for (let i = 0; i < 7; i += 1) {
    const slice = document.createElement("span");
    slice.style.setProperty("--i", i);
    stack.appendChild(slice);
  }
  return stack;
}

function tensorSizeClass(dims) {
  const numbers = String(dims).match(/\d+/g) || [];
  const spatial = Number(numbers[numbers.length - 1]);
  if (spatial >= 56) return "tensor-56";
  if (spatial >= 28) return "tensor-28";
  if (spatial >= 14) return "tensor-14";
  return "tensor-7";
}

function renderNeatArrow() {
  return createText("div", "neat-arrow", "");
}

function renderHeadPanel() {
  const panel = createText("section", "bp-panel head-panel", "");
  panel.appendChild(createText("div", "bp-panel-title", "分类头"));
  panel.appendChild(createText("div", "bp-op-text", "avgpool\n全局平均池化"));
  panel.appendChild(renderMiniStack("feature tiny", "layer4", el.layerSelect.value, "[512, 1, 1]"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "flatten\n展平"));
  panel.appendChild(createText("div", "bp-vector", "512"));
  panel.appendChild(renderDownArrow());
  panel.appendChild(createText("div", "bp-op-text", "fc: 全连接层\n权重 [1000, 512]"));
  panel.appendChild(createText("div", "bp-vector", "1000"));
  panel.appendChild(createText("div", "bp-dims strong", "输出：[1000]"));
  return panel;
}

function renderMiniStack(kind, layerId, selectedLayer, dims) {
  const stack = document.createElement("button");
  stack.type = "button";
  stack.className = `bp-stack ${kind}`;
  if (layerId) {
    stack.classList.add("clickable");
    stack.addEventListener("click", () => selectLayerAndRun(layerId));
  }
  if (layerId && layerId === selectedLayer) stack.classList.add("active");

  for (let i = 0; i < 7; i += 1) {
    const slice = document.createElement("span");
    slice.style.setProperty("--i", i);
    stack.appendChild(slice);
  }
  if (dims) stack.appendChild(createText("em", "bp-stack-dims", dims));
  return stack;
}

function renderDownArrow() {
  return createText("div", "bp-down-arrow", "");
}

function renderPanelArrow() {
  return createText("div", "bp-panel-arrow", "");
}

function renderAddNode() {
  return createText("div", "bp-add", "+");
}

function renderLegend() {
  const legend = createText("div", "bp-legend", "");
  legend.appendChild(renderLegendItem("feature", "特征图\n多通道特征张量 [C, H, W]"));
  legend.appendChild(renderLegendItem("kernel", "卷积核 / 卷积核权重\n[out_channels, in_channels, k, k]"));
  legend.appendChild(renderLegendItem("arrow", "主分支流"));
  legend.appendChild(renderLegendItem("skip", "恒等映射 identity skip"));
  legend.appendChild(renderLegendItem("down", "下采样分支 downsample 1 x 1 conv"));
  return legend;
}

function renderLegendItem(type, text) {
  const item = createText("div", `bp-legend-item ${type}`, "");
  if (type === "feature") item.appendChild(renderMiniStack("feature legend-stack", "", "", ""));
  if (type === "kernel") item.appendChild(renderMiniStack("kernel legend-stack", "", "", ""));
  if (type === "arrow") item.appendChild(createText("span", "legend-arrow-main", ""));
  if (type === "skip") item.appendChild(createText("span", "legend-arrow-skip", ""));
  if (type === "down") item.appendChild(createText("span", "legend-arrow-down", ""));
  item.appendChild(createText("span", "bp-legend-text", text));
  return item;
}

function renderSceneNode(node, selectedLayer) {
  const nodeEl = document.createElement("button");
  nodeEl.type = "button";
  nodeEl.className = `scene-node ${node.type} ${node.size || ""}`;
  if (node.kernel) {
    if (node.kernel.includes("7,7")) nodeEl.classList.add("k7");
    if (node.kernel.includes("3,3")) nodeEl.classList.add("k3");
    if (node.kernel.includes("1,1")) nodeEl.classList.add("k1");
  }
  if (node.layer) nodeEl.classList.add("clickable");
  if (node.layer === selectedLayer) nodeEl.classList.add("active");
  if (node.layer) {
    nodeEl.addEventListener("click", () => selectLayerAndRun(node.layer));
  }

  nodeEl.appendChild(createText("div", "scene-top-label", node.title));

  const block = document.createElement("div");
  block.className = "scene-block";

  if (node.type === "image") {
    block.appendChild(createText("div", "image-plane-text", "RGB"));
  } else if (node.type.includes("kernel")) {
    block.appendChild(createText("div", "kernel-mark", node.kernel.includes("1,1") ? "1x1" : node.kernel.includes("7,7") ? "7x7" : "3x3"));
  } else if (node.type === "pool") {
    block.appendChild(createText("div", "pool-mark", "pool"));
  } else if (node.type === "add") {
    block.appendChild(createText("div", "add-mark", "+"));
  } else if (node.type === "fc") {
    block.appendChild(createText("div", "fc-line", ""));
    block.appendChild(createText("div", "fc-line", ""));
    block.appendChild(createText("div", "fc-line short", ""));
  } else {
    for (let i = 0; i < (node.depth || 3); i += 1) {
      const slice = document.createElement("span");
      slice.className = "feature-slice";
      slice.style.setProperty("--slice-index", i);
      block.appendChild(slice);
    }
  }

  nodeEl.appendChild(block);
  if (node.kernel) nodeEl.appendChild(createText("div", "scene-kernel", `卷积核 ${node.kernel}`));
  if (node.out) nodeEl.appendChild(createText("div", "scene-meta", node.out));
  nodeEl.appendChild(createText("div", "scene-dims", node.dims || ""));
  return nodeEl;
}

function selectLayerAndRun(layerId) {
  const option = Array.from(el.layerSelect.options).find((item) => item.value === layerId);
  if (!option) return;
  el.layerSelect.value = layerId;
  renderNetworkDiagram();
  runCnn();
}

function renderKernel(kernel) {
  if (!kernel) {
    el.kernelPanel.classList.add("hidden");
    el.kernelGrid.innerHTML = "";
    return;
  }

  const rows = kernel.length;
  const cols = kernel[0].length;
  el.kernelGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  el.kernelGrid.innerHTML = "";

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = createText("div", "kernel-cell", Number(kernel[r][c]).toFixed(3).replace(/\.?0+$/, ""));
      el.kernelGrid.appendChild(cell);
    }
  }

  el.kernelPanel.classList.remove("hidden");
}

function setLoading(loadingEl, buttonEl, active) {
  loadingEl.classList.toggle("hidden", !active);
  buttonEl.disabled = active;
}

function setOptionalImage(imageEl, src) {
  if (!imageEl) return;
  if (src) {
    imageEl.src = src;
  } else {
    imageEl.removeAttribute("src");
  }
}

function clearOptionalImage(imageEl) {
  if (imageEl) imageEl.removeAttribute("src");
}

function syncClassicalSourceImage() {
  if (!el.classicalSourceImage || !el.sourcePreview) return;
  if (el.sourcePreview.src) {
    el.classicalSourceImage.src = el.sourcePreview.src;
  } else {
    el.classicalSourceImage.removeAttribute("src");
  }
}

function renderTopChannels(items) {
  if (!el.topChannelGrid) return;
  el.topChannelGrid.innerHTML = "";

  if (!items.length) {
    el.topChannelGrid.appendChild(createText("div", "top-channel-empty", "生成特征图后显示。"));
    return;
  }

  items.forEach((item) => {
    const card = createText("article", "top-channel-card", "");
    const header = createText("div", "top-channel-header", "");
    header.appendChild(createText("strong", "", `Top-${item.rank} 通道`));
    header.appendChild(createText("span", "", `ch ${Number(item.channel_index) + 1}`));

    const img = document.createElement("img");
    img.src = item.image;
    img.alt = `Top-${item.rank} 通道特征图`;

    const stats = createText("dl", "top-channel-stats", "");
    appendStat(stats, "平均激活值", formatMetric(item.mean_activation));
    appendStat(stats, "最大激活值", formatMetric(item.max_activation));
    appendStat(stats, "激活区域占比", formatPercent(item.active_ratio));

    card.appendChild(header);
    card.appendChild(img);
    card.appendChild(stats);
    el.topChannelGrid.appendChild(card);
  });
}

function appendStat(list, label, value) {
  list.appendChild(createText("dt", "", label));
  list.appendChild(createText("dd", "", value));
}

function formatMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "未知";
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "未知";
}

function createText(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

init();
