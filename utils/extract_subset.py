#coding=utf-8
import struct
import os
from PIL import Image
import shutil
from multiprocessing import Pool, cpu_count
import uuid

# ================= 配置区域 =================
#这是你的GNT数据源路径，请修改为你实际的路径
DATA_PATH = r"d:\project6\cnwordextract\HWDB1.1\train" 
# 这是提取后的小数据集存放路径
OUTPUT_PATH = r"d:\project6\cnwordextract\HWDB1.1\subset_631"
WORKER_COUNT = 4  # 进程数

# 631个常用汉字列表 (已清洗)
target_chars_str = """
一是了我的
不人在他有这个上们来到时
大地为子中你说生国年着就那和要她出也得里后自以会
家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开
美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法
间斯知世什两次使身者被高已亲其进此话常与活正感
见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并别
真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死
安写性马光白或住难望教命花结乐色
更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带
万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求
清王化空业思切怎非找片罗钱紶吗语元喜曾离飞科言干流欢约各即指合反题必该论交终
林请医晚制球决窢传画保读运及则房早院量苦火布品近坐产答星精视五连司巴
奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造
谈容极随演收首根讲整式取照办强石古华諣拿计您装似足双妻尼转诉米称丽客南领节衣
站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗
独复痛消社算
算义竟确酒需单治卡幸兰念举仅钟怕 共毛句息功官待究跟穿室易游程号居考突皮哪费
倒价图具刚脑永歌 响商礼细专黄块脚味灵改据般破 引食仍存众注笔甚某沉血备习校默
务土微娘须试怀料调广蜖苏显赛查密议底列富梦 错座参八除跑亮假印设线 温虽掉京初
养香停际 致阳纸李纳验助激够严证帝饭忘 趣支
"""

# 去除换行和空格，生成字符集合用于快速查找
target_chars = set(target_chars_str.replace("\n", "").replace(" ", ""))

print(f"目标提取字符数: {len(target_chars)}")
# ===========================================

def ensure_dir_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def process_single_gnt(args):
    """
    单个GNT文件的处理函数，用于多进程调用
    """
    file_path, output_path, whitelist_set = args
    file_name = os.path.basename(file_path)
    
    local_extracted_count = 0
    
    print(f"[开始] 进程正在处理: {file_name}")
    
    try:
        with open(file_path, "rb") as f:
            while True:
                # 读取头部检测是否还有数据
                header_bytes = f.read(4)
                if not header_bytes:
                    break
                
                try:
                    tag_code = f.read(2)
                    try:
                        char = tag_code.decode('gbk')
                    except UnicodeDecodeError:
                        width = struct.unpack('<h', f.read(2))[0]
                        height = struct.unpack('<h', f.read(2))[0]
                        f.read(width * height)
                        continue

                    width = struct.unpack('<h', f.read(2))[0]
                    height = struct.unpack('<h', f.read(2))[0]
                    bitmap_data = f.read(width * height)
                    
                    if char in whitelist_set:
                        im = Image.frombytes('L', (width, height), bitmap_data)
                        
                        char_dir = os.path.join(output_path, char)
                        
                        # 使用UUID作为文件名，避免多进程写入同一文件夹时的命名冲突
                        # 这样我们就不需要进程必须等待文件锁，速度最快
                        safe_filename = f"{file_name}_{uuid.uuid4().hex[:8]}.jpg"
                        
                        im.save(os.path.join(char_dir, safe_filename))
                        
                        local_extracted_count += 1
                        
                except Exception as inner_e:
                    print(f"文件 {file_name} 读取出错: {inner_e}")
                    break
    except Exception as e:
        print(f"打开文件 {file_name} 失败: {e}")
        
    print(f"[完成] {file_name}: 提取了 {local_extracted_count} 张图片")
    return local_extracted_count

def main():
    # 0. 预先创建好所有目标文件夹
    print("正在预创建文件夹...")
    ensure_dir_exists(OUTPUT_PATH)
    for char in target_chars:
        ensure_dir_exists(os.path.join(OUTPUT_PATH, char))

    # 1. 收集文件
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据路径 {DATA_PATH} 不存在。")
        return

    gnt_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.gnt')]
    print(f"找到 {len(gnt_files)} 个GNT文件。启动 {WORKER_COUNT} 个进程进行提取...")
    
    # 2. 准备任务参数
    tasks = [(f, OUTPUT_PATH, target_chars) for f in gnt_files]

    # 3. 并行执行
    total_extracted = 0
    with Pool(processes=WORKER_COUNT) as pool:
        for count in pool.imap_unordered(process_single_gnt, tasks):
            total_extracted += count
            
    print(f"所有任务完成！共提取 {total_extracted} 张图片。")

    # 4. 生成索引文件
    print("正在生成 file_list.txt 和 char_dict.txt ...")
    with open("file_list.txt", "w", encoding='utf-8') as f_list, \
         open("char_dict.txt", "w", encoding='utf-8') as f_dict:
        
        existing_chars = [d for d in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, d))]
        label_map = {char: idx for idx, char in enumerate(existing_chars)}
        
        for char, idx in label_map.items():
            f_dict.write(f"{char} {idx}\n")
            
        file_count = 0
        for char in existing_chars:
            char_dir = os.path.join(OUTPUT_PATH, char)
            try:
                # 使用 scandir 更快
                with os.scandir(char_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith('.jpg'):
                            line = f"{char}/{entry.name} {label_map[char]}\n"
                            f_list.write(line)
                            file_count += 1
            except OSError:
                pass
                        
    print(f"索引生成完毕。最终有效图片数: {file_count}, 类别数: {len(existing_chars)}")

if __name__ == '__main__':
    main()
