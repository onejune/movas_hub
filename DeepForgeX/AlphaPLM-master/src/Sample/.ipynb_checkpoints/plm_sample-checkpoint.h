#ifndef PLM_SAMPLE_H_
#define PLM_SAMPLE_H_

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>

using namespace std;

class plm_sample
{
public:
    int y;
    vector<pair<string, double> > x;
    plm_sample(const string& line);
    plm_sample(const string& line, const vector<string>& column_names, const unordered_map<string, bool>& selected_features, int label_index);
    plm_sample(const string& line, 
                  const vector<string>& column_names, 
                  const vector<vector<string>>& combine_schema_feature, 
                  int label_index);
        void generate_combinations(const vector<string>& components, 
                  const vector<vector<string>>& component_value_lists,
                  size_t index, string current_key, 
                  vector<string>& combined_keys);
private:
    static const string spliter;
    static const string innerSpliter;
    static const unordered_set<string> invalid_values; // 预定义无效值集合
};

const string plm_sample::spliter = " ";
const string plm_sample::innerSpliter = ":";
const unordered_set<string> plm_sample::invalid_values = {"-1.0", "-", ""};

plm_sample::plm_sample(const string& line)
{
    this->x.clear();
    size_t posb = line.find_first_not_of(spliter, 0);
    size_t pose = line.find_first_of(spliter, posb);
    int label = atoi(line.substr(posb, pose-posb).c_str());
    this->y = label > 0 ? 1 : -1;
    string key;
    double value;
    while(pose < line.size())
    {
        posb = line.find_first_not_of(spliter, pose);
        if(posb == string::npos)
        {
            break;
        }
        pose = line.find_first_of(innerSpliter, posb);
        if(pose == string::npos)
        {
            cerr << "wrong line of sample input\n" << line << endl;
            exit(1);
        }
        key = line.substr(posb, pose-posb);
        posb = pose + 1;
        if(posb >= line.size())
        {
            cerr << "wrong line of sample input\n" << line << endl;
            exit(1);
        }
        pose = line.find_first_of(spliter, posb);
        value = stod(line.substr(posb, pose-posb));
        if(value != 0)
        {
            this->x.push_back(make_pair(key, value));
        }
    }
}

// 新的构造函数，用于处理 CSV 格式的输入
plm_sample::plm_sample(const string& line, 
                     const vector<string>& column_names, 
                     const unordered_map<string, bool>& selected_features, int label_index)
{
    this->x.clear();
    stringstream ss(line);
    string token;
    int column_index = 0;

    while (getline(ss, token, '\002'))
    {
        if (column_index == label_index) {
            int label = atoi(token.c_str());
            this->y = label > 0 ? 1 : -1;
        } else if (column_index < column_names.size()) {
            string feature_name = column_names[column_index];
            if (selected_features.find(feature_name) == selected_features.end()) {
                column_index++;
                continue;
            }
            if (feature_name != "tags" && feature_name != "bundle_list") { //单值特征
                if (invalid_values.find(token) != invalid_values.end()) {
                    column_index++;
                    continue;
                }
                string key = feature_name + "=" + token;
                this->x.emplace_back(key, 1.0);
            }
            else {
                stringstream value_ss;
                value_ss.str(token);
                value_ss.clear();

                string sub_token;
                string key_prefix = feature_name + "="; // 缓存前缀
                while (getline(value_ss, sub_token, '\001')) {
                    if (invalid_values.find(sub_token) != invalid_values.end()) {
                        continue;
                    }

                    string key;
                    key.reserve(key_prefix.size() + sub_token.size()); // 预分配内存
                    key = key_prefix;
                    key += sub_token;
                    this->x.emplace_back(key, 1.0); // 避免拷贝
                }
            }
        }
        column_index++;
    }
}

// 新的构造函数，用于处理 CSV 格式的输入并支持组合特征
plm_sample::plm_sample(const string& line, const vector<string>& column_names, 
                     const vector<vector<string>>& combine_schema_feature, 
                     int label_index)
{
    this->x.clear();
    stringstream ss(line);
    string token;
    int column_index = 0;

    // 存储所有列的值，用于后续组合特征的拼接
    unordered_map<string, vector<string>> column_values; // 修改为存储多值特征

    // 读取 CSV 行
    while (getline(ss, token, '\002'))
    {
        if (column_index == label_index) {
            // 解析 label
            int label = atoi(token.c_str());
            this->y = label > 0 ? 1 : -1;
        } else if (column_index < column_names.size()) {
            // 解析特征
            string feature_name = column_names[column_index];
            // 处理多值特征
            vector<string> feature_values;
            if (feature_name != "tags") { //单值特征
                if (invalid_values.find(token) == invalid_values.end()) {
                    feature_values.push_back(token);
                }
            }
            else {    
                stringstream value_ss(token);
                string sub_token;
                while (getline(value_ss, sub_token, '\001')) {
                    if (invalid_values.find(token) == invalid_values.end()) {
                        feature_values.push_back(sub_token);
                    }
                }
            }
            // 将当前列的值存储到 column_values 中
            column_values[feature_name] = feature_values;
        }
        column_index++;
    }

    // 处理组合特征
    for (const auto& components : combine_schema_feature) {

        // 检查所有组合特征的组成部分是否存在且有效
        bool all_components_valid = true;
        vector<vector<string>> component_value_lists;
        for (const string& component : components) {
            if (column_values.find(component) == column_values.end() || 
                column_values[component].empty()) {
                all_components_valid = false;
                break;
            }
            component_value_lists.push_back(column_values[component]);
        }

        // 如果所有组成部分都有效，则构造组合特征
        if (all_components_valid) {
            // 枚举所有可能的组合
            vector<string> combined_keys;
            generate_combinations(components, component_value_lists, 0, "", combined_keys);

            // 添加组合特征
            for (const string& combined_key : combined_keys) {
                this->x.push_back(make_pair(combined_key, 1.0));
            }
        }
    }
}

// 辅助函数：生成所有可能的组合特征
void plm_sample::generate_combinations(const vector<string>& components, 
                                      const vector<vector<string>>& component_value_lists,
                                      size_t index, string current_key, 
                                      vector<string>& combined_keys)
{
    if (index == components.size()) {
        combined_keys.push_back(current_key);
        return;
    }

    for (const string& value : component_value_lists[index]) {
        string new_key = current_key.empty() ? components[index] + "=" + value : 
                       current_key + "#" + components[index] + "=" + value;
        generate_combinations(components, component_value_lists, index + 1, new_key, combined_keys);
    }
}

#endif