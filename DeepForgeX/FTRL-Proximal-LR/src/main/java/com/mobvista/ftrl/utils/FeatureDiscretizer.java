package com.mobvista.ftrl.utils;

public class FeatureDiscretizer {
    
    public enum DiscretizeMethod {
        LOG,           // 对数离散化
        MANUAL_BUCKET, // 手动分桶
        EQUAL_WIDTH    // 等宽分桶
    }
    
    public static String discretize(String value, String featureName, DiscretizeMethod method) {
        if (value == null || value.isEmpty()) {
            return value;
        }
        
        try {
            double numValue = Double.parseDouble(value);
            switch (method) {
                case LOG:
                    return String.valueOf(Math.log1p(numValue));
                case MANUAL_BUCKET:
                    return String.valueOf(manualBucket(numValue, featureName));
                case EQUAL_WIDTH:
                    return String.valueOf(equalWidthBucket(numValue, 10)); // 默认10个桶
                default:
                    return value;
            }
        } catch (NumberFormatException e) {
            return value;
        }
    }

    public static int discretize_bucket(String value) {
        if (value == null || value.isEmpty()) {
            return -1;
        }
        try {
            int cnt = Integer.parseInt(value);
            if (cnt <= 10) return cnt;
            if (cnt <= 15) return 15;
            if (cnt <= 20) return 20;
            if (cnt <= 25) return 25;
            return 100;
        } catch (NumberFormatException e) {
            return -1;
        }
    }
    
    private static int manualBucket(double value, String featureName) {
        switch (featureName) {
            case "cate_list":
                return cateListBucket((int)value);
            case "price":
                return priceBucket(value);
            // 可以添加更多特征的分桶逻辑
            default:
                return (int)value;
        }
    }
    
    private static int cateListBucket(int cnt) {
        if (cnt <= 3) return 3;
        if (cnt <= 5) return 5;
        if (cnt <= 10) return 10;
        if (cnt <= 15) return 15;
        if (cnt <= 20) return 20;
        if (cnt <= 25) return 25;
        return 100;
    }
    
    private static int priceBucket(double price) {
        if (price <= 10) return 10;
        if (price <= 50) return 50;
        if (price <= 100) return 100;
        if (price <= 500) return 500;
        return 1000;
    }
    
    private static int equalWidthBucket(double value, int numBuckets) {
        // 简单的等宽分桶，可以根据需要调整
        int bucket = (int)(value / 100 * numBuckets);
        return Math.min(bucket, numBuckets - 1);
    }
}