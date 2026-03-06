# OSS 上传指南

将 SID Pipeline 打包并上传到阿里云 OSS 的完整步骤。

---

## 第一步：打包代码

在 DSW 终端执行：

```bash
cd /mnt/workspace/MiniOneRec/rq/pipeline
chmod +x package.sh
bash package.sh
```

这会生成：
- `/mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_YYYYMMDD_HHMMSS.tar.gz`（带时间戳）
- `/mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_latest.tar.gz`（最新版本）

---

## 第二步：上传到 OSS

### 方法1：使用 ossutil（推荐）

```bash
# 1. 安装 ossutil（如果还没装）
wget http://gosspublic.alicdn.com/ossutil/1.7.15/ossutil64
chmod 755 ossutil64

# 2. 配置 ossutil（首次使用）
./ossutil64 config
# 按提示输入：
#   - Endpoint: oss-cn-hangzhou.aliyuncs.com（根据你的区域选择）
#   - AccessKeyID: 你的 AccessKey ID
#   - AccessKeySecret: 你的 AccessKey Secret

# 3. 上传文件
./ossutil64 cp /mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_latest.tar.gz \
  oss://your-bucket-name/projects/sid_pipeline/

# 4. 设置公共读权限（可选，如果需要公开下载）
./ossutil64 set-acl oss://your-bucket-name/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz public-read
```

### 方法2：使用 DSW 文件管理器

1. 在 DSW 左侧文件树找到 `/mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_latest.tar.gz`
2. 右键 → 下载到本地
3. 登录阿里云 OSS 控制台：https://oss.console.aliyun.com/
4. 选择 Bucket → 上传文件
5. 上传完成后，点击文件 → 获取 URL

### 方法3：使用 Python SDK

```python
import oss2

# 配置
auth = oss2.Auth('your-access-key-id', 'your-access-key-secret')
bucket = oss2.Bucket(auth, 'oss-cn-hangzhou.aliyuncs.com', 'your-bucket-name')

# 上传
local_file = '/mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_latest.tar.gz'
oss_path = 'projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz'

bucket.put_object_from_file(oss_path, local_file)
print(f"上传成功！")

# 获取下载链接
url = bucket.sign_url('GET', oss_path, 3600*24*7)  # 7天有效期
print(f"下载链接: {url}")
```

---

## 第三步：获取下载链接

### 公共读链接（永久有效）

```
https://your-bucket-name.oss-cn-hangzhou.aliyuncs.com/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz
```

### 私有链接（带签名，有时效）

```bash
# 使用 ossutil 生成
./ossutil64 sign oss://your-bucket-name/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz \
  --timeout 604800  # 7天有效期（秒）
```

---

## 第四步：提供给老师的信息

### 邮件/消息模板

```
老师您好，

SID 生成 Pipeline 已打包完成并上传到 OSS：

📦 下载链接：
https://your-bucket-name.oss-cn-hangzhou.aliyuncs.com/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz

📄 使用说明：
1. 下载并解压：
   wget <上面的链接>
   tar -xzf sid_pipeline_v1.0_latest.tar.gz
   cd pipeline

2. 修改配置：
   vi config.sh
   # 填入数据路径、模型路径

3. 执行流程：
   bash step3_text2emb.sh       # 文本→Embedding
   bash step4_train_rqvae.sh    # 训练RQ-VAE
   bash step5_generate_sid.sh   # 生成SID
   bash step7_generate_summary.sh  # 汇总表格
   bash step8_evaluate_quality.sh  # 质量评估

详细文档见压缩包内的 README.md 和 使用说明.md

📊 功能特性：
- 支持 CPU/GPU 一键切换
- 8个步骤模块化执行
- 自动生成汇总表格（CSV格式，Excel可打开）
- 自动评估SID质量（碰撞率、语义分组准确率）
- 所有配置集中在 config.sh，修改方便

如有问题请随时联系。
```

---

## 常用 OSS 命令参考

```bash
# 查看 bucket 内容
./ossutil64 ls oss://your-bucket-name/projects/sid_pipeline/

# 下载文件
./ossutil64 cp oss://your-bucket-name/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz ./

# 删除文件
./ossutil64 rm oss://your-bucket-name/projects/sid_pipeline/old_version.tar.gz

# 查看文件信息
./ossutil64 stat oss://your-bucket-name/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz
```

---

## 注意事项

1. **AccessKey 安全**：不要把 AccessKey 写在代码里或提交到 git
2. **权限设置**：如果是内部使用，建议设置为私有；公开分享才设置公共读
3. **文件大小**：打包后约 50-100KB（不含模型和数据）
4. **有效期**：公共读链接永久有效；签名链接需要设置有效期
5. **区域选择**：选择离你最近的 OSS 区域，下载更快

---

## 快速命令（复制粘贴）

```bash
# 一键打包并上传（修改 bucket 名称）
cd /mnt/workspace/MiniOneRec/rq/pipeline
bash package.sh
./ossutil64 cp /mnt/workspace/MiniOneRec/packages/sid_pipeline_v1.0_latest.tar.gz \
  oss://your-bucket-name/projects/sid_pipeline/
./ossutil64 set-acl oss://your-bucket-name/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz public-read

# 获取下载链接
echo "https://your-bucket-name.oss-cn-hangzhou.aliyuncs.com/projects/sid_pipeline/sid_pipeline_v1.0_latest.tar.gz"
```
