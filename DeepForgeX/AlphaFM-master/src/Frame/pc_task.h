#ifndef PC_TASK_H
#define PC_TASK_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>

using namespace std;

class pc_task
{
public:
    pc_task(){}
    virtual void run_task(vector<string>& dataBuffer) = 0;
    inline void load_feature_config();

public:
    vector<string> column_names;
    unordered_map<string, bool> selected_features;
    vector<vector<string>> combine_fea_list;
    int label_index = -1;  // 记录 label 列的位置
};

inline void pc_task::load_feature_config(){
    // 读取 column_name.txt
    ifstream column_file("./conf/column_name");
    string line;
    while (getline(column_file, line))
    {
        // 移除可能存在的 '\r' 字符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // 删除最后一个字符
        }
        // 使用 stringstream 分割字符串
        stringstream ss(line);
        string first_column, second_column;

        if (ss >> first_column >> second_column) { // 提取第一列和第二列
            column_names.push_back(second_column);
            //cout << second_column << " Length: " << second_column.length() << endl;
            if (second_column == "label") {
                label_index = column_names.size() - 1; // 记录 label 列的位置
            }
        }
    }
    column_file.close();
    cout << "load column_name:" << column_names.size() << " label_index:" << label_index << endl;
 
    // 读取 combine_schema.txt
    ifstream schema_file("./conf/combine_schema");
    while (getline(schema_file, line)) {
        // 忽略以 '#' 开头的行
        if (!line.empty() && line[0] == '#') {
            continue;
        }
        
        // 移除可能存在的 '\r' 字符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // 删除最后一个字符
        }

        stringstream ss(line); // 使用 stringstream 分割
        string feature;
        
        vector<string> fea_list;
        while (getline(ss, feature, '#')) { // 按 '#' 分割
            if (!feature.empty()) { // 忽略空字符串
                selected_features[feature] = true;
                fea_list.push_back(feature);
            }
        }
        combine_fea_list.push_back(fea_list);
    }
    schema_file.close();
    cout << "load single features:" << selected_features.size() << endl;
    cout << "load combined features:" << combine_fea_list.size() << endl;
    //for (auto it = selected_features.begin(); it != selected_features.end(); ++it) {
    //    std::cout << it->first << std::endl;
    //}
}

#endif //PC_TASK_H
