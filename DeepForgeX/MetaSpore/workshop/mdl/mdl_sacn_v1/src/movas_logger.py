import os, sys, time, traceback, inspect, random
from datetime import datetime, timedelta
from inspect import getframeinfo, stack
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession

'''
模块的功能：收集日志内容并写入文件；
 df._jdf.showString(10, 100, False)可以使用变量捕获 df.show()的结果;
'''

#计算函数执行时间
def how_much_time(func):
    def wrapper(*args, **kw):  
        start_time = time.time()
        rtn = func(*args, **kw)
        end_time = time.time()
        dur_min = round((end_time - start_time) / 60, 4)
        time_info = '[%s] time_use: %s minutes.' % (func.__name__, dur_min)
        MovasLogger.add_log(level = 'time_use', content = time_info)
        return rtn
    return wrapper 

class LogUnit():
    def __init__(self, caption, content, level, log_time, caller):
        self.level = level
        self.caption = caption
        self.content = content
        self.log_time = log_time
        self.caller = caller
    def format(self):
        filename = self.caller.filename.split('/')[-1]
        s = '[%s][%s][%s:%s:%s] %s' % (self.log_time, self.level, filename, self.caller.lineno, self.caller.function, self.content)
        #s = '%s' % (self.content)
        return s

class MovasLogger():
    output_path = ''
    log_unit_list = []
    log_time_use = []
    spark = None
    debug_mode = False
    log_rate = 0.0001
        
    @staticmethod
    def init(spark, output_path):
        MovasLogger.output_path = output_path
        MovasLogger.log_unit_list = []
        MovasLogger.log_time_use = []
        MovasLogger.spark = spark
        MovasLogger.debug_mode = False
        
    @staticmethod
    def init_spark():
        MovasLogger.spark = SparkSession.builder \
            .appName("MovasLogger") \
            .getOrCreate()
            
    @staticmethod
    def set_debug_mode(mode):
        MovasLogger.debug_mode = mode
    
    @staticmethod
    def get_call_stack_str(skip=2):
        """返回调用栈字符串，skip 表示跳过顶部多少帧（默认跳过自身）"""
        lines = []
        for frame_info in inspect.stack()[skip:]:
            if frame_info.function == 'wrapper':
                continue
            line = f"File: {frame_info.filename}, Line: {frame_info.lineno}, Function: {frame_info.function}"
            lines.append(line)
        header = "\n==================== Call Stack =====================\n"
        tail = "\n====================================================="
        return header + "\n".join(lines) + tail

    @staticmethod
    def add_log(level = 'INFO', caption = 'none', content = 'none'):
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        try:
            #caller_func = traceback.extract_stack()[-2][2]
            caller = getframeinfo(stack()[1][0])
        except:
            caller = None
        log_unit = LogUnit(caption, content, level, time_now, caller)
        if level == 'time_use':
            MovasLogger.log_time_use.append(log_unit)
        else:
            MovasLogger.log_unit_list.append(log_unit)
        print("[movas_log]", log_unit.format())
    
    @staticmethod
    def log(*args):
        if len(args) == 1 and isinstance(args[0], list):
            content = ','.join(str(x) for x in args[0])
        else:
            content = ' '.join(str(x) for x in args)
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        try:
            caller = getframeinfo(stack()[1][0])
        except:
            caller = None
        log_unit = LogUnit("none", content, "INFO", time_now, caller)
        MovasLogger.log_unit_list.append(log_unit)
        print("[movas_log]", log_unit.format())

    @staticmethod
    def debug(*args):
        if MovasLogger.debug_mode == False:
            return
        if len(args) == 1 and isinstance(args[0], list):
            content = ','.join(str(x) for x in args[0])
        else:
            content = ' '.join(str(x) for x in args)
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        try:
            #caller_func = traceback.extract_stack()[-2][2]
            caller = getframeinfo(stack()[1][0])
        except:
            caller = None
        log_unit = LogUnit("none", content, "DEBUG", time_now, caller)
        MovasLogger.log_unit_list.append(log_unit)
        print("[movas_log]", log_unit.format())

    @staticmethod
    def random_log(content):
        if random.randint(0, 100) < MovasLogger.log_rate:
            MovasLogger.log(content)
    
    @staticmethod
    def debug_call_stack():
        MovasLogger.add_log(level = 'DEBUG', content = MovasLogger.get_call_stack_str())

    #保存 log 至本地，适用于 local 模式
    @staticmethod
    def save_to_local():
        log_str = '\n'.join(MovasLogger.get_log_str_list())
        print(log_str)
        print('write log to %s' % MovasLogger.output_path)
        fout = open(MovasLogger.output_path, 'a')
        fout.write(log_str)
        fout.close()
    
    #返回 log list，用于构建 rdd 然后存储至 s3，适用于分布式模式
    @staticmethod
    def get_log_str_list():
        r_list = []
        for unit in MovasLogger.log_unit_list:
            r_list.append(unit.format())
        for unit in MovasLogger.log_time_use:
            r_list.append(unit.format())
        return r_list
    
    #捕获 df.show()的结果
    @staticmethod
    def get_df_showString(df, lines = 20):
        s = 'debug_mode = false'
        if MovasLogger.debug_mode == True:
            s = df._jdf.showString(lines, 100, False)
        return s
    
    @staticmethod
    def save_to_s3():
        #rdd = MovasLogger.spark.parallelize(MovasLogger.get_log_str_list(), 1)
        #rdd.saveAsTextFile(MovasLogger.output_path)
        MovasLogger.add_log(content = "log count = %d" % len(MovasLogger.get_log_str_list()))
        if MovasLogger.spark is None:
            MovasLogger.init_spark()
        df = MovasLogger.spark.createDataFrame(MovasLogger.get_log_str_list(), StringType())
        df.show()
        df.repartition(1).write.csv(MovasLogger.output_path, mode="overwrite")
