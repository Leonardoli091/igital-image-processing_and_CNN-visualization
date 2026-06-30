from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class FilterResult:
    image: Image.Image
    title: str
    description: str
    kernel: Optional[List[List[float]]] = None


CLASSICAL_OPERATIONS = [
    {
        "id": "grayscale",
        "label": "灰度化",
        "category": "灰度与点运算",
        "principle": "按人眼对红、绿、蓝亮度的敏感程度进行加权，把 RGB 三通道压缩成单通道灰度图，保留主要明暗结构。",
    },
    {
        "id": "invert",
        "label": "反色变换",
        "category": "灰度与点运算",
        "principle": "对每个像素执行 255 - value，把亮区域变暗、暗区域变亮，用来观察互补颜色和反相细节。",
    },
    {
        "id": "brightness_contrast",
        "label": "亮度 / 对比度增强",
        "category": "灰度增强",
        "principle": "使用线性灰度变换 value * a + b，同步调整像素差异和整体明暗，让图像更清楚。",
    },
    {
        "id": "contrast_stretch",
        "label": "对比度拉伸",
        "category": "灰度增强",
        "principle": "把图像中主要灰度范围重新映射到 0 到 255，扩展动态范围，增强低对比度图像的层次。",
    },
    {
        "id": "gamma_correction",
        "label": "伽马校正",
        "category": "灰度增强",
        "principle": "使用幂函数进行非线性亮度调整；gamma 小于 1 会提升暗部响应，gamma 大于 1 会压暗亮度。",
    },
    {
        "id": "log_transform",
        "label": "对数变换",
        "category": "灰度增强",
        "principle": "用对数函数压缩高亮区域、拉开暗部灰度差异，适合观察暗部细节或动态范围较大的图像。",
    },
    {
        "id": "hist_equalization",
        "label": "直方图均衡",
        "category": "灰度增强",
        "principle": "根据灰度累计分布函数重新分配像素值，使灰度分布更均匀，从而提升整体对比度。",
    },
    {
        "id": "rgb_hist_equalization",
        "label": "彩色亮度均衡",
        "category": "灰度增强",
        "principle": "只对彩色图像的亮度信息做均衡，再保持原有色彩比例，减少直接均衡 RGB 通道造成的偏色。",
    },
    {
        "id": "mean_denoise",
        "label": "均值滤波去噪",
        "category": "平滑去噪",
        "principle": "用邻域像素平均值替代中心像素，能削弱随机噪声，但也会让边缘和纹理变得更平滑。",
    },
    {
        "id": "median_denoise",
        "label": "中值滤波去噪",
        "category": "平滑去噪",
        "principle": "取邻域像素的中位数作为输出，对椒盐噪声很有效，同时比均值滤波更容易保留边缘。",
    },
    {
        "id": "gaussian_denoise",
        "label": "高斯滤波去噪",
        "category": "平滑去噪",
        "principle": "按高斯权重对邻域加权平均，中心像素权重大、远处权重小，常用于自然平滑和降噪。",
    },
    {
        "id": "motion_blur",
        "label": "运动模糊",
        "category": "平滑去噪",
        "principle": "沿固定方向做平均卷积，模拟相机或目标运动时产生的方向性拖影。",
    },
    {
        "id": "sharpen",
        "label": "锐化",
        "category": "锐化与边缘检测",
        "principle": "通过增强中心像素和削弱邻域像素来放大灰度突变，使边缘、纹理和轮廓更明显。",
    },
    {
        "id": "unsharp_mask",
        "label": "反锐化掩蔽",
        "category": "锐化与边缘检测",
        "principle": "先得到平滑图，再用原图减去平滑图得到细节层，最后把细节叠回原图以增强清晰度。",
    },
    {
        "id": "emboss",
        "label": "浮雕效果",
        "category": "锐化与边缘检测",
        "principle": "用带方向性的卷积核强调一侧亮、一侧暗的灰度变化，产生类似光照下凸起的浮雕观感。",
    },
    {
        "id": "sobel",
        "label": "Sobel 边缘",
        "category": "锐化与边缘检测",
        "principle": "分别计算水平和垂直方向的一阶梯度，并合成梯度幅值，突出边缘且带有一定平滑效果。",
    },
    {
        "id": "prewitt",
        "label": "Prewitt 边缘",
        "category": "锐化与边缘检测",
        "principle": "用简单的一阶差分模板估计水平和垂直梯度，计算速度快，适合展示基础边缘检测思想。",
    },
    {
        "id": "roberts",
        "label": "Roberts 边缘",
        "category": "锐化与边缘检测",
        "principle": "用 2x2 交叉差分检测对角方向的灰度突变，对细小边缘敏感，但也更容易受噪声影响。",
    },
    {
        "id": "laplacian",
        "label": "Laplacian 边缘",
        "category": "锐化与边缘检测",
        "principle": "使用二阶微分检测灰度变化的突变位置，能够同时响应多个方向的边缘。",
    },
    {
        "id": "high_pass",
        "label": "高通滤波",
        "category": "锐化与边缘检测",
        "principle": "抑制低频平滑区域、保留高频细节和边缘，因此常用于纹理增强和轮廓提取。",
    },
    {
        "id": "otsu_threshold",
        "label": "Otsu 二值化",
        "category": "阈值分割",
        "principle": "自动寻找让前景和背景类间方差最大的阈值，把灰度图分成黑白两类。",
    },
    {
        "id": "adaptive_threshold",
        "label": "自适应阈值",
        "category": "阈值分割",
        "principle": "为每个局部区域单独计算阈值，能处理光照不均的图像，比全局阈值更灵活。",
    },
    {
        "id": "morph_erode",
        "label": "形态学腐蚀",
        "category": "形态学处理",
        "principle": "用结构元素扫描二值图，只有邻域满足条件才保留前景，可收缩目标并去除小白噪点。",
    },
    {
        "id": "morph_dilate",
        "label": "形态学膨胀",
        "category": "形态学处理",
        "principle": "只要结构元素覆盖到前景就扩展前景区域，可填补小空洞、连接断裂区域。",
    },
    {
        "id": "morph_open",
        "label": "形态学开运算",
        "category": "形态学处理",
        "principle": "先腐蚀后膨胀，用于去除小前景噪点，同时尽量保持较大目标的形状。",
    },
    {
        "id": "morph_close",
        "label": "形态学闭运算",
        "category": "形态学处理",
        "principle": "先膨胀后腐蚀，用于填补小孔洞和细小裂缝，让目标区域更连贯。",
    },
    {
        "id": "posterize",
        "label": "色彩量化",
        "category": "色彩与风格化",
        "principle": "减少每个颜色通道可用的灰度级数，把连续色彩压缩成少量色阶，形成分层的色彩效果。",
    },
]


CLASSICAL_OPERATION_POINTS = {
    "grayscale": [
        "计算公式：Y = 0.299R + 0.587G + 0.114B，绿色权重最大，因为人眼对绿色亮度更敏感。",
        "输出只保留亮度信息，不再保留颜色信息，后续边缘检测、阈值分割通常先转灰度。",
    ],
    "invert": [
        "逐像素计算：out = 255 - in，原来接近白色的像素会变黑，接近黑色的像素会变白。",
        "该变换不改变物体边界位置，只改变灰度和颜色方向，常用于突出暗背景中的亮结构。",
    ],
    "brightness_contrast": [
        "本项目使用线性变换：out = 1.18 * in + 16，其中 1.18 控制对比度，16 控制整体亮度。",
        "像素差异会被放大，图像整体也会变亮；超过 255 的值会被截断，所以高亮区域可能饱和。",
    ],
    "contrast_stretch": [
        "先取 2% 和 98% 分位灰度作为低、高端点，再把这个区间线性映射到 0 到 255。",
        "这样可以避开少量极暗或极亮异常点，让主体灰度范围被充分拉开。",
    ],
    "gamma_correction": [
        "计算公式：out = 255 * (in / 255)^gamma，本项目 gamma = 0.65。",
        "gamma 小于 1 时，暗部像素会被提升得更多，因此暗部纹理会更容易看见。",
    ],
    "log_transform": [
        "计算公式：out = log(1 + in) / log(256) * 255，低灰度段增长快，高灰度段增长慢。",
        "因此它会压缩亮部、提升暗部，适合动态范围很大或暗部信息较多的图像。",
    ],
    "hist_equalization": [
        "先统计灰度直方图，再计算累计分布函数 CDF，最后用 CDF 把旧灰度映射到新灰度。",
        "目标是让灰度分布更均匀，低对比度图像会变清楚，但噪声也可能被一起增强。",
    ],
    "rgb_hist_equalization": [
        "不是分别均衡 R、G、B，而是先估计亮度并均衡亮度，再按比例缩放彩色通道。",
        "这样能提升亮度对比度，同时尽量保持原图色相关系，减少明显偏色。",
    ],
    "mean_denoise": [
        "使用 3x3 均值卷积核，每个邻域权重都是 1/9，输出为周围 9 个像素的平均值。",
        "随机噪声会被平均掉，但边缘两侧的像素也会互相混合，所以图像会变模糊。",
    ],
    "median_denoise": [
        "在 3x3 邻域内排序，取中间值作为当前像素输出，而不是求平均。",
        "孤立的黑点或白点通常位于排序两端，会被中位数替换，所以适合去除椒盐噪声。",
    ],
    "gaussian_denoise": [
        "使用 5x5 高斯权重核，中心权重大，距离中心越远权重越小。",
        "相比均值滤波，它平滑更自然，对边缘的破坏通常更小，常作为边缘检测前的预处理。",
    ],
    "motion_blur": [
        "本项目使用 9x9 对角线平均核，只有对角线方向有权重，相当于沿一个方向累积像素。",
        "会形成方向性拖影，用来模拟相机抖动或物体运动造成的模糊。",
    ],
    "sharpen": [
        "使用卷积核 [[0,-1,0],[-1,5,-1],[0,-1,0]]，中心像素被放大，四邻域被减去。",
        "平坦区域变化不大，灰度突变处会被强化，所以轮廓和纹理看起来更清晰。",
    ],
    "unsharp_mask": [
        "先用高斯核得到模糊图，再计算细节层 detail = 原图 - 模糊图。",
        "最后把细节层加回原图，相当于只增强高频细节，而不是简单整体加亮。",
    ],
    "emboss": [
        "使用带方向性的正负卷积核，一侧像素被加权为亮，另一侧被加权为暗。",
        "灰度变化会被转成类似光照阴影的效果，因此边缘呈现凸起或凹陷的浮雕感。",
    ],
    "sobel": [
        "分别用 Sobel Gx、Gy 模板计算水平和垂直方向梯度，再合成 magnitude = sqrt(Gx^2 + Gy^2)。",
        "Sobel 模板中间行或列权重为 2，带有轻微平滑作用，所以抗噪声能力比简单差分更好。",
    ],
    "prewitt": [
        "使用 Prewitt Gx、Gy 模板估计一阶梯度，每行或每列权重相同。",
        "它比 Sobel 更简单，能显示基础梯度边缘，但对噪声的抑制略弱。",
    ],
    "roberts": [
        "使用 2x2 交叉差分核，计算两个对角方向的灰度差。",
        "因为邻域很小，它对细小边缘响应敏感，但图像有噪声时也更容易产生误检。",
    ],
    "laplacian": [
        "使用二阶差分核 [[0,-1,0],[-1,4,-1],[0,-1,0]]，检测灰度变化率突变的位置。",
        "它不区分具体方向，水平、垂直等多方向边缘都会响应，但二阶微分对噪声也敏感。",
    ],
    "high_pass": [
        "使用高通核 [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]，中心像素与周围 8 个像素作差。",
        "低频平滑背景会被抵消，高频边缘、纹理和噪声会保留下来并被增强。",
    ],
    "otsu_threshold": [
        "遍历所有可能阈值 T，把像素分成背景和前景两类，并计算两类之间的类间方差。",
        "选择类间方差最大的 T，表示此时前景和背景分离最明显，然后输出黑白二值图。",
    ],
    "adaptive_threshold": [
        "本项目对每个像素计算 21x21 局部均值，并使用 local_mean - 5 作为该位置阈值。",
        "每个区域有自己的阈值，所以当图像左亮右暗或光照不均时，比全局阈值更稳定。",
    ],
    "morph_erode": [
        "先用 Otsu 得到二值图，再用 3x3 结构元素检查邻域，只有邻域都满足前景条件才保留白色。",
        "白色区域会向内收缩，小白点会消失，目标边界也会变细。",
    ],
    "morph_dilate": [
        "先用 Otsu 得到二值图，再用 3x3 结构元素扩展前景，只要邻域有前景就输出白色。",
        "白色区域会向外扩张，可连接细小断裂，也可能让目标变粗。",
    ],
    "morph_open": [
        "开运算 = 先腐蚀再膨胀；腐蚀先去掉小白噪点，膨胀再尽量恢复主体大小。",
        "适合去除比结构元素还小的前景噪声，同时保留较大目标。",
    ],
    "morph_close": [
        "闭运算 = 先膨胀再腐蚀；膨胀先填补小黑洞和缝隙，腐蚀再恢复边界大小。",
        "适合让断裂的前景区域更连通，也能填充目标内部的小孔洞。",
    ],
    "posterize": [
        "本项目把每个颜色通道压缩成 4 个等级：先按 step = 256 / 4 分桶，再取每个桶的中心值。",
        "连续色彩会变成有限色阶，细腻渐变减少，图像呈现分块、海报化效果。",
    ],
}


def list_classical_operations():
    operations = []
    for item in CLASSICAL_OPERATIONS:
        operation = dict(item)
        operation["principle_points"] = CLASSICAL_OPERATION_POINTS.get(item["id"], [])
        operations.append(operation)
    return operations


def apply_classical_operation(image: Image.Image, operation: str) -> FilterResult:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

    if operation == "grayscale":
        gray = rgb_to_gray(rgb)
        return FilterResult(to_pil(gray), "灰度化", "按加权亮度公式将 RGB 图像转换为灰度图。")

    if operation == "invert":
        out = 255 - rgb
        return FilterResult(to_pil(out), "反色变换", "将每个像素映射为 255 - value，突出互补色结构。")

    if operation == "brightness_contrast":
        out = rgb * 1.18 + 16
        return FilterResult(to_pil(out), "亮度 / 对比度增强", "使用线性变换 value * 1.18 + 16 同时提升亮度和对比度。")

    if operation == "contrast_stretch":
        out = contrast_stretch(rgb, low_percentile=2, high_percentile=98)
        return FilterResult(to_pil(out), "对比度拉伸", "将 2% 到 98% 分位的像素范围线性拉伸到 0 到 255。")

    if operation == "gamma_correction":
        gamma = 0.65
        out = 255.0 * np.power(np.clip(rgb / 255.0, 0, 1), gamma)
        return FilterResult(to_pil(out), f"伽马校正 γ={gamma}", "非线性增强暗部细节，gamma 小于 1 时图像整体变亮。")

    if operation == "log_transform":
        out = np.log1p(rgb) / np.log(256.0) * 255.0
        return FilterResult(to_pil(out), "对数变换", "压缩高亮区域并提升暗部响应，适合观察暗部细节。")

    if operation == "mean_denoise":
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        out = convolve_rgb(rgb, kernel)
        return FilterResult(to_pil(out), "均值滤波去噪", "使用 3x3 平均卷积核平滑图像。", kernel.tolist())

    if operation == "gaussian_denoise":
        kernel = np.array(
            [
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1, 4, 7, 4, 1],
            ],
            dtype=np.float32,
        )
        kernel = kernel / kernel.sum()
        out = convolve_rgb(rgb, kernel)
        return FilterResult(to_pil(out), "高斯滤波去噪", "使用 5x5 高斯卷积核降低随机噪声。", rounded_kernel(kernel))

    if operation == "median_denoise":
        out = median_filter_rgb(rgb, size=3)
        return FilterResult(to_pil(out), "中值滤波去噪", "用邻域中值替换当前像素，适合抑制椒盐噪声。")

    if operation == "motion_blur":
        kernel = np.eye(9, dtype=np.float32) / 9.0
        out = convolve_rgb(rgb, kernel)
        return FilterResult(to_pil(out), "运动模糊", "使用对角线方向的 9x9 核模拟相机或物体运动造成的拖影。", rounded_kernel(kernel))

    if operation == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = convolve_rgb(rgb, kernel)
        return FilterResult(to_pil(out), "锐化", "增强中心像素并削弱周围像素，突出细节。", kernel.tolist())

    if operation == "unsharp_mask":
        blur_kernel = np.array(
            [
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1, 4, 7, 4, 1],
            ],
            dtype=np.float32,
        )
        blur_kernel = blur_kernel / blur_kernel.sum()
        blurred = convolve_rgb(rgb, blur_kernel)
        out = rgb + 1.4 * (rgb - blurred)
        return FilterResult(to_pil(out), "反锐化掩蔽", "先高斯模糊，再把原图与模糊图的差值加回原图以增强边缘。")

    if operation == "emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        out = convolve_rgb(rgb, kernel) + 128
        return FilterResult(to_pil(out), "浮雕效果", "用方向性卷积突出斜向灰度变化，并加偏置形成浮雕视觉效果。", kernel.tolist())

    if operation == "sobel":
        gray = rgb_to_gray(rgb)
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sx = convolve_gray(gray, gx)
        sy = convolve_gray(gray, gy)
        magnitude = np.sqrt(sx * sx + sy * sy)
        return FilterResult(to_pil(normalize_uint8(magnitude)), "Sobel 边缘", "分别计算水平和垂直梯度，再合成边缘强度图。", gx.tolist())

    if operation == "prewitt":
        gray = rgb_to_gray(rgb)
        gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        sx = convolve_gray(gray, gx)
        sy = convolve_gray(gray, gy)
        magnitude = np.sqrt(sx * sx + sy * sy)
        return FilterResult(to_pil(normalize_uint8(magnitude)), "Prewitt 边缘", "使用 Prewitt 算子估计水平和垂直方向的一阶梯度。", gx.tolist())

    if operation == "roberts":
        gray = rgb_to_gray(rgb)
        gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        sx = convolve_gray(gray, gx)
        sy = convolve_gray(gray, gy)
        magnitude = np.sqrt(sx * sx + sy * sy)
        return FilterResult(to_pil(normalize_uint8(magnitude)), "Roberts 边缘", "使用 2x2 交叉差分算子检测细小斜向边缘。", gx.tolist())

    if operation == "laplacian":
        gray = rgb_to_gray(rgb)
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        out = np.abs(convolve_gray(gray, kernel))
        return FilterResult(to_pil(normalize_uint8(out)), "Laplacian 边缘", "利用二阶差分突出灰度突变区域。", kernel.tolist())

    if operation == "high_pass":
        gray = rgb_to_gray(rgb)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        out = np.abs(convolve_gray(gray, kernel))
        return FilterResult(to_pil(normalize_uint8(out)), "高通滤波", "保留高频细节与边缘，抑制平滑背景区域。", kernel.tolist())

    if operation == "hist_equalization":
        gray = rgb_to_gray(rgb).astype(np.uint8)
        out = histogram_equalization(gray)
        return FilterResult(to_pil(out), "直方图均衡", "重新分配灰度级，提高整体对比度。")

    if operation == "rgb_hist_equalization":
        out = equalize_luminance(rgb)
        return FilterResult(to_pil(out), "彩色亮度均衡", "只均衡亮度通道，尽量保留原图色彩关系。")

    if operation == "otsu_threshold":
        gray = rgb_to_gray(rgb).astype(np.uint8)
        threshold = otsu_threshold(gray)
        out = np.where(gray >= threshold, 255, 0).astype(np.uint8)
        return FilterResult(to_pil(out), f"Otsu 二值化 T={threshold}", "自动寻找类间方差最大的阈值。")

    if operation == "adaptive_threshold":
        gray = rgb_to_gray(rgb).astype(np.float32)
        local_mean = box_filter_gray(gray, size=21)
        out = np.where(gray >= local_mean - 5, 255, 0).astype(np.uint8)
        return FilterResult(to_pil(out), "自适应阈值", "以局部均值为阈值，适合光照不均的图像二值化。")

    if operation == "morph_erode":
        binary = binary_otsu(rgb)
        out = morph_binary(binary, "erode", size=3)
        return FilterResult(to_pil(out), "形态学腐蚀", "收缩白色前景区域，可去除小亮点。")

    if operation == "morph_dilate":
        binary = binary_otsu(rgb)
        out = morph_binary(binary, "dilate", size=3)
        return FilterResult(to_pil(out), "形态学膨胀", "扩张白色前景区域，可填补小断裂。")

    if operation == "morph_open":
        binary = binary_otsu(rgb)
        out = morph_binary(morph_binary(binary, "erode", size=3), "dilate", size=3)
        return FilterResult(to_pil(out), "形态学开运算", "先腐蚀后膨胀，用于去除小的前景噪点。")

    if operation == "morph_close":
        binary = binary_otsu(rgb)
        out = morph_binary(morph_binary(binary, "dilate", size=3), "erode", size=3)
        return FilterResult(to_pil(out), "形态学闭运算", "先膨胀后腐蚀，用于填补前景中的小孔洞。")

    if operation == "posterize":
        levels = 4
        step = 256 // levels
        out = np.floor(rgb / step) * step + step / 2
        return FilterResult(to_pil(out), "色彩量化", "将每个颜色通道压缩到 4 个等级，形成分块化色彩效果。")

    raise ValueError("未知的图像处理操作。")


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def convolve_rgb(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    channels = []
    for index in range(3):
        channels.append(convolve_gray(rgb[:, :, index], kernel))
    return np.stack(channels, axis=2)


def convolve_gray(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge").astype(np.float32)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)

    for r in range(kh):
        for c in range(kw):
            out += kernel[r, c] * padded[r : r + h, c : c + w]
    return out


def median_filter_rgb(rgb: np.ndarray, size: int = 3) -> np.ndarray:
    channels = []
    for index in range(3):
        channels.append(median_filter_gray(rgb[:, :, index], size))
    return np.stack(channels, axis=2)


def median_filter_gray(gray: np.ndarray, size: int = 3) -> np.ndarray:
    pad = size // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2))


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for value in gray.reshape(-1):
        hist[int(value)] += 1

    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return gray

    cdf_min = nonzero[0]
    total = gray.size
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.clip(round((cdf[i] - cdf_min) / max(1, total - cdf_min) * 255), 0, 255)
    return lut[gray]


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.zeros(256, dtype=np.int64)
    for value in gray.reshape(-1):
        hist[int(value)] += 1

    total = gray.size
    total_sum = sum(i * hist[i] for i in range(256))
    count0 = 0
    sum0 = 0.0
    best_threshold = 0
    best_score = -1.0

    for t in range(256):
        count0 += hist[t]
        sum0 += t * hist[t]
        count1 = total - count0
        if count0 == 0 or count1 == 0:
            continue

        mean0 = sum0 / count0
        mean1 = (total_sum - sum0) / count1
        score = count0 * count1 * (mean0 - mean1) * (mean0 - mean1)
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold


def contrast_stretch(rgb: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[:, :, channel]
        low = float(np.percentile(plane, low_percentile))
        high = float(np.percentile(plane, high_percentile))
        if high - low < 1e-6:
            out[:, :, channel] = plane
        else:
            out[:, :, channel] = (plane - low) / (high - low) * 255.0
    return out


def equalize_luminance(rgb: np.ndarray) -> np.ndarray:
    luminance = rgb_to_gray(rgb).astype(np.uint8)
    equalized = histogram_equalization(luminance).astype(np.float32)
    scale = equalized / np.maximum(luminance.astype(np.float32), 1.0)
    return rgb * scale[:, :, None]


def box_filter_gray(gray: np.ndarray, size: int = 21) -> np.ndarray:
    pad = size // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge").astype(np.float32)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return summed / float(size * size)


def binary_otsu(rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(rgb).astype(np.uint8)
    threshold = otsu_threshold(gray)
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def morph_binary(binary: np.ndarray, operation: str, size: int = 3) -> np.ndarray:
    pad = size // 2
    padded = np.pad(binary, ((pad, pad), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    if operation == "erode":
        return windows.min(axis=(-1, -2)).astype(np.uint8)
    if operation == "dilate":
        return windows.max(axis=(-1, -2)).astype(np.uint8)
    raise ValueError("Unknown morphology operation.")


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-6:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - low) / (high - low) * 255).astype(np.uint8)


def to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def rounded_kernel(kernel: np.ndarray) -> List[List[float]]:
    return [[round(float(value), 4) for value in row] for row in kernel]
