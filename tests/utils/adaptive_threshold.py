import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class AdaptiveThreshold:
    """
    自适应阈值调整方法类，用于增强异常检测的鲁棒性
    
    提供多种阈值计算策略：
    1. 多参数混合策略 (混合多种统计量)
    2. 核密度估计策略 (基于概率密度)
    3. 高斯混合模型 (GMM) 策略
    4. 聚类分离策略
    5. 动态自适应策略 (考虑时间窗口内的分布变化)
    """
    
    def __init__(self, strategy='hybrid', window_size=None, alpha=0.05, n_components=2):
        """
        初始化自适应阈值计算器
        
        参数:
            strategy: 阈值计算策略，可选值: 'hybrid', 'kde', 'gmm', 'cluster', 'dynamic'
            window_size: 用于动态策略的滑动窗口大小
            alpha: 异常判定的置信度
            n_components: GMM或聚类的组件数量
        """
        self.strategy = strategy
        self.window_size = window_size
        self.alpha = alpha
        self.n_components = n_components
        self.history = []
    
    def compute_threshold(self, train_scores, test_scores=None, anomaly_ratio=None):
        """
        根据选定的策略计算阈值
        
        参数:
            train_scores: 训练集上的异常分数
            test_scores: 测试集上的异常分数 (可选)
            anomaly_ratio: 期望的异常比例 (可选)
            
        返回:
            threshold: 计算得到的阈值
        """
        if self.strategy == 'hybrid':
            return self._hybrid_threshold(train_scores, anomaly_ratio)
        elif self.strategy == 'kde':
            return self._kde_threshold(train_scores)
        elif self.strategy == 'gmm':
            return self._gmm_threshold(train_scores)
        elif self.strategy == 'cluster':
            return self._cluster_threshold(train_scores)
        elif self.strategy == 'dynamic':
            combined = np.concatenate([train_scores, test_scores]) if test_scores is not None else train_scores
            return self._dynamic_threshold(combined)
        elif self.strategy == 'bootstrap':
            return self._bootstrap_threshold(train_scores)
        elif self.strategy == 'trimmed':
            return self._trimmed_threshold(train_scores)
        elif self.strategy == 'local_gradient':
            return self._local_gradient_threshold(train_scores)
        elif self.strategy == 'entropy_weighted':
            return self._entropy_weighted_threshold(train_scores)
        else:
            # 默认使用百分位数方法
            if anomaly_ratio is None:
                anomaly_ratio = 0.05  # 默认异常比例为5%
            return np.percentile(train_scores, 100 - 100 * anomaly_ratio)
    
    def _hybrid_threshold(self, scores, anomaly_ratio=None):
        """
        混合策略：结合多种统计量的阈值计算方法
        
        结合平均值、标准差、中位数、IQR和分位数等多种统计量
        对不同的统计方法进行加权融合，增强阈值的鲁棒性
        
        参数:
            scores: 异常分数
            anomaly_ratio: 期望的异常比例
            
        返回:
            threshold: 计算得到的阈值
        """
        if anomaly_ratio is None:
            anomaly_ratio = 0.05

        # 计算各种统计量
        mean = np.mean(scores)
        std = np.std(scores)
        median = np.median(scores)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        percentile = np.percentile(scores, 100 * (1 - anomaly_ratio))
        
        # Z分数对应的阈值 (假设正态分布)
        z_threshold = mean + stats.norm.ppf(1 - anomaly_ratio) * std
        
        # IQR方法的阈值
        iqr_threshold = q75 + 1.5 * iqr
        
        # 基于中位数绝对偏差(MAD)的阈值
        mad = np.median(np.abs(scores - median))
        mad_threshold = median + stats.norm.ppf(1 - anomaly_ratio) * mad * 1.4826  # 常数1.4826用于使MAD成为标准差的一致估计
        
        # 加权融合多种阈值 (权重可以根据实际效果调整)
        weights = np.array([0.2, 0.2, 0.3, 0.3])  # 分别对应z_threshold, iqr_threshold, mad_threshold, percentile
        threshold = weights.dot(np.array([z_threshold, iqr_threshold, mad_threshold, percentile]))
        
        return threshold
    
    def _kde_threshold(self, scores):
        """
        核密度估计策略：基于概率密度估计的阈值计算方法
        
        使用KDE估计分数的概率分布，然后找到密度快速下降的点作为阈值
        
        参数:
            scores: 异常分数
            
        返回:
            threshold: 计算得到的阈值
        """
        # 标准化分数
        scores = scores.reshape(-1, 1)
        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(scores)
        
        # 使用核密度估计
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(scores_scaled)
        
        # 在一个范围内计算概率密度
        x_grid = np.linspace(np.min(scores_scaled), np.max(scores_scaled) * 1.2, 1000).reshape(-1, 1)
        log_pdf = kde.score_samples(x_grid)
        pdf = np.exp(log_pdf)
        
        # 计算密度的梯度
        pdf_grad = np.gradient(pdf)
        
        # 找到密度梯度最大的下降点 (从高到低找第一个梯度值低于阈值的点)
        # 我们希望捕捉到的是分布"尾部"的开始位置
        grad_threshold = np.percentile(pdf_grad, 10)  # 使用梯度的低分位数作为阈值
        
        # 从密度高的区域开始扫描
        max_density_idx = np.argmax(pdf)
        threshold_idx = None
        
        for i in range(max_density_idx, len(pdf_grad) - 1):
            if pdf_grad[i] < grad_threshold and pdf[i] < pdf[max_density_idx] * 0.1:
                threshold_idx = i
                break
        
        if threshold_idx is None:
            # 如果未找到合适的阈值点，回退到使用分位数
            threshold = np.percentile(scores, 95)
        else:
            # 将阈值转换回原始比例
            threshold_scaled = x_grid[threshold_idx]
            threshold = scaler.inverse_transform(threshold_scaled)[0][0]
        
        return threshold
    
    def _gmm_threshold(self, scores):
        """
        高斯混合模型策略：假设分数来自多个高斯分布
        
        使用GMM将分数聚类为正常和异常两类，然后计算决策边界
        
        参数:
            scores: 异常分数
            
        返回:
            threshold: 计算得到的阈值
        """
        # 重塑分数为适合GMM的形状
        X = scores.reshape(-1, 1)
        
        # 拟合GMM模型
        gmm = GaussianMixture(n_components=self.n_components, random_state=42)
        gmm.fit(X)
        
        # 获取各组件的参数
        means = gmm.means_.flatten()
        weights = gmm.weights_
        covariances = gmm.covariances_.flatten()
        
        # 识别正常组件和异常组件 (假设正常分数较低，异常分数较高)
        normal_idx = np.argmin(means)
        anomaly_idx = np.argmax(means)
        
        # 假设正常分数和异常分数都服从高斯分布，计算两个分布的交叉点
        normal_mean = means[normal_idx]
        normal_std = np.sqrt(covariances[normal_idx])
        anomaly_mean = means[anomaly_idx]
        anomaly_std = np.sqrt(covariances[anomaly_idx])
        
        # 解二次方程求交点
        a = 1/(2*anomaly_std**2) - 1/(2*normal_std**2)
        b = normal_mean/(normal_std**2) - anomaly_mean/(anomaly_std**2)
        c = anomaly_mean**2/(2*anomaly_std**2) - normal_mean**2/(2*normal_std**2) - np.log(normal_std/anomaly_std)
        
        # 求解判别点
        if a == 0:
            threshold = -c/b
        else:
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                # 如果没有交点，回退到使用正常分布的上尾部
                threshold = normal_mean + 2 * normal_std
            else:
                # 选择两个正态分布之间的交点
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                # 选择在两个均值之间的交点
                if x1 > normal_mean and x1 < anomaly_mean:
                    threshold = x1
                elif x2 > normal_mean and x2 < anomaly_mean:
                    threshold = x2
                else:
                    # 回退到正常分布的上尾部
                    threshold = normal_mean + 2 * normal_std
        
        return threshold
    
    def _cluster_threshold(self, scores):
        """
        聚类分离策略：使用K-means聚类将分数分为正常和异常
        
        参数:
            scores: 异常分数
            
        返回:
            threshold: 计算得到的阈值
        """
        # 重塑分数为适合K-means的形状
        X = scores.reshape(-1, 1)
        
        # 应用K-means聚类
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # 获取聚类中心
        centers = kmeans.cluster_centers_.flatten()
        
        # 确定哪个聚类代表正常 (假设正常分数较低)
        normal_center = min(centers)
        anomaly_center = max(centers)
        
        # 生成正常样本和异常样本的掩码
        normal_mask = (clusters == np.argmin(centers))
        
        # 获取正常样本
        normal_scores = scores[normal_mask]
        
        # 计算正常样本的上界作为阈值
        normal_upper = np.mean(normal_scores) + 3 * np.std(normal_scores)
        
        # 计算两个聚类中心之间的中点
        midpoint = (normal_center + anomaly_center) / 2
        
        # 将两种阈值结合
        threshold = (normal_upper + midpoint) / 2
        
        return threshold
    
    def _dynamic_threshold(self, scores):
        """
        动态自适应策略：考虑时间序列的变化趋势
        
        使用滑动窗口来捕获数据分布的变化，动态调整阈值
        
        参数:
            scores: 异常分数
            
        返回:
            threshold: 计算得到的阈值
        """
        # 更新历史数据
        self.history.append(scores)
        
        # 如果历史数据超过窗口大小，保留最近的数据
        if self.window_size is not None and len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # 获取当前窗口中的所有数据
        window_data = np.concatenate(self.history)
        
        # 计算基本统计量
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        # 计算自适应阈值 (使用均值加上标准差的倍数)
        # Z分数可以根据期望的假阳性率(FPR)来调整
        z_score = stats.norm.ppf(1 - self.alpha)
        threshold = mean + z_score * std
        
        # 对最近数据的趋势进行分析，如果有上升趋势，考虑提高阈值
        if len(self.history) >= 2:
            recent_mean = np.mean(self.history[-1])
            previous_mean = np.mean(self.history[-2])
            trend_factor = max(0, (recent_mean - previous_mean) / previous_mean)
            threshold = threshold * (1 + trend_factor)
        
        return threshold

    def _bootstrap_threshold(self, scores, n_iterations=100, percentile=95):
        """使用 bootstrap 重采样计算阈值，该方法通过多次采样得到稳定的上界分位数"""
        boot_values = []
        for i in range(n_iterations):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            boot_values.append(np.percentile(sample, percentile))
        return np.mean(boot_values)

    def _trimmed_threshold(self, scores, trim_fraction=0.1, factor=1.5):
        """基于截断统计量的方法：先剔除极端值后计算均值和标准差，阈值为均值加上 factor 倍标准差"""
        lower = np.percentile(scores, trim_fraction * 100)
        upper = np.percentile(scores, 100 - trim_fraction * 100)
        trimmed_scores = scores[(scores >= lower) & (scores <= upper)]
        trimmed_mean = np.mean(trimmed_scores)
        trimmed_std = np.std(trimmed_scores)
        return trimmed_mean + factor * trimmed_std

    def _local_gradient_threshold(self, scores, percentile=95, sensitivity=0.5):
        """基于局部梯度的阈值方法
        
        首先计算基准分位数阈值，然后根据高分区域的局部梯度大小来调整阈值。
        局部梯度大的区域意味着异常变化快速，应该提高阈值的灵敏度。
        
        参数:
            scores: 异常分数
            percentile: 基础分位数 (默认95)
            sensitivity: 梯度调整系数 (0-1)，控制对局部变化的敏感程度
            
        返回:
            threshold: 基于局部梯度信息调整后的阈值
        """
        # 计算基础分位数阈值
        base_threshold = np.percentile(scores, percentile)
        
        # 计算分数序列的梯度
        gradients = np.abs(np.gradient(np.sort(scores)))
        
        # 提取高分区域的梯度 (高于80%分位数的部分)
        high_score_idx = int(len(scores) * 0.8)
        local_gradients = gradients[high_score_idx:]
        
        # 计算局部梯度均值，并进行归一化
        if len(local_gradients) > 0:
            mean_local_gradient = np.mean(local_gradients)
            # 归一化到[0,1]区间，用于调整
            norm_factor = max(np.max(gradients), 1e-8)  # 避免除零
            norm_gradient = mean_local_gradient / norm_factor
            
            # 根据局部梯度调整阈值
            # 当局部梯度大时，提高阈值以更好地捕获突变
            adjustment = 1.0 + sensitivity * norm_gradient
            adjusted_threshold = base_threshold * adjustment
            
            return adjusted_threshold
        
        return base_threshold  # 如果没有足够的数据，则返回基础阈值
    
    def _entropy_weighted_threshold(self, scores, percentile=95, bins=20, max_weight=0.5):
        """基于信息熵加权的阈值方法
        
        计算异常分数分布的信息熵，作为不确定性的度量，并用其调整阈值。
        当分布熵值高（分布分散）时，适当提高阈值以减少假阳性。
        
        参数:
            scores: 异常分数
            percentile: 基础分位数 (默认95)
            bins: 计算熵时使用的直方图分箱数
            max_weight: 熵权重的最大调整系数
            
        返回:
            threshold: 基于信息熵调整后的阈值
        """
        # 计算基础分位数阈值
        base_threshold = np.percentile(scores, percentile)
        
        # 计算分数的直方图和概率分布
        hist, bin_edges = np.histogram(scores, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        prob = hist * bin_width  # 转换为概率，确保总和为1
        prob = prob[prob > 0]  # 移除零概率，避免log(0)
        
        # 计算分布的信息熵
        entropy = -np.sum(prob * np.log2(prob))
        
        # 将熵归一化到[0,1]区间
        # 对于均匀分布，熵最大为log2(bins)
        max_entropy = np.log2(bins)
        norm_entropy = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0
        
        # 根据熵调整阈值
        # 当熵高（分布分散）时，增加阈值以降低假阳性率
        adjustment = 1.0 + max_weight * norm_entropy
        adjusted_threshold = base_threshold * adjustment
        
        return adjusted_threshold

def get_adaptive_threshold(train_scores, test_scores=None, strategy='hybrid', anomaly_ratio=None, window_size=None, alpha=0.05):
    """
    便捷函数：获取自适应阈值
    
    参数:
        train_scores: 训练集上的异常分数
        test_scores: 测试集上的异常分数 (可选)
        strategy: 阈值计算策略
        anomaly_ratio: 预期的异常比例
        window_size: 动态策略的窗口大小
        alpha: 异常判定的置信度
        
    返回:
        threshold: 计算得到的阈值
    """
    threshold_calculator = AdaptiveThreshold(strategy=strategy, window_size=window_size, alpha=alpha)
    return threshold_calculator.compute_threshold(train_scores, test_scores, anomaly_ratio) 