
import pytest
import os
import subprocess
import time
import json
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy, generate_proxy_endpoints
from run_vllm_mock import strart_vllm_mock, cleanup_subprocess
from port_manager import find_free_port,load_ports

from collections import defaultdict
import port_manager
import requests
import concurrent.futures
import random

# Configuration
PREFILL_NUM = 4
DECODE_NUM = 4
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../../../omni/accelerators/sched/omni_proxy/omni_proxy.sh"
APP_START_MARKER = "Application startup complete."
STARTUP_TIMEOUT = 120  # seconds
model_path=f"{CUR_DIR}/mock_model/"
COVRC_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../")
TOP_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../../")

@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    processes = strart_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)

def setup_proxy_basic(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    ports = load_ports(PREFILL_NUM, DECODE_NUM)
    prefill_list = generate_proxy_endpoints(ports["prefill"])
    decode_list = generate_proxy_endpoints(ports["decode"])
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_balance.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_balance.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_balance.log",
            "--stream-ops", "add",
            # "--omni-proxy-model-path", f"{CUR_DIR}/mock_model",
            # "--omni-proxy-schedule-algo", "earliest_batch",
            "--no-reuseport",
            "--keepalive-nginx"
        ]
        # print(f"\n[SETUP] Starting proxy for load balance check with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        return 1

def setup_proxy_earliest(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    ports = load_ports(PREFILL_NUM, DECODE_NUM)
    prefill_list = generate_proxy_endpoints(ports["prefill"])
    decode_list = generate_proxy_endpoints(ports["decode"])
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_balance.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_balance.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_balance.log",
            "--stream-ops", "add",
            # "--omni-proxy-model-path", f"{CUR_DIR}/mock_model",
            "--omni-proxy-schedule-algo", "earliest_batch",
            "--no-reuseport",
            "--keepalive-nginx"
        ]
        # print(f"\n[SETUP] Starting proxy for load balance check with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        return 1

def fetch_post(url, headers, data):
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)  
        return {
            "url": url,
            "status": response.status_code,
            "text": response.text[:200] + "..." if len(response.text) > 200 else response.text
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "error": str(e)
        }

def teardown_proxy_balance():
    try:
        cmd = "kill -QUIT $(ps -ef --sort=-lstart | grep 'nginx: master' | grep -v grep | head -n1 | awk '{print $2}')"
        
        print(f"\n[TEARDOWN] Stopping proxy with command: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[TEARDOWN] Script succeeded. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Teardown script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)

def test_chat_completions_with_proxy_basic(setup_teardown):

    proxy_port = find_free_port()
    ret = setup_proxy_basic(proxy_port)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"  


    data = [
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:232
            "messages": [{"role": "user", "content": "童大海心地却好，叫道：小心！上前伸手欲扶。他那知这人有意在群英之前显一手上乘武功，手掌刚搭上那人左臂，那人一勾一带，施出了大擒拿手中一招倒跌金刚。童大海身不由主的向台外直飞出去，砰的一声，结结实实的摔在地下。众人瞧那人时，但见他衣饰修洁，长眉俊目，原来是郭靖的弟子武修文。郭靖坐在台左第一排椅上，见他这招大擒拿手虽巧妙洒脱，但行径轻狂，大违忠厚之道，心下不悦，脸色便沉了下来。果然台下有多人不服，台东台西同时响起了三个声音，叫道：好俊功夫，兄弟来领教几招！这算什么？人家好意扶你，你却施暗算！发话声中，三个人同时跃上台来。武修文学兼郭靖、黄蓉两家，且家学渊源，得父亲与师叔授了一阳指神技，在后辈英雄中已算第一流人才，见三人齐至，暗暗欢喜，寻思：我同时败此三人，方显得功夫。反而怕这三人分别来斗，更不说话，身形晃动，剎时之间向上台三人每人发了一招。那三人尚未站稳，敌招却倏忽已至，忙举手招架。武修文不待对方缓过手来，双掌翻飞，竟以一围三，将三个对手包围在核心，自己占了外势。那三人互相挤撞，拳脚难以施展。群雄相顾失色，均想：郭大侠名震当世，果然名不虚传，连教出来的徒儿也这般厉害？那三个人互相不识，不知旁人的武功拳路，遭武修文一围住，没法呼应照顾，反而各自牵制。三人连冲数次，始终抢不出武修文以绵密掌法构成的包围圈子。完颜萍在台下见丈夫已稳占上风，心中欢喜。郭芙却道：这三个人脓包，当然不是小武哥哥的敌手。其实他何必这时候便逞英雄，耗费了力气？待会真有高手上台，岂不难以抵敌？完颜萍微笑不语。耶律燕平时极爱和郭芙斗口，嫡亲姑嫂，互不相让，这时早猜中了嫂子的心意，说道：小叔叔先上去收拾一批，待他不成了，敦儒又上去收拾一批。他又不成了，我哥哥这才上台，独败群雄，让你安安稳稳的做个帮主夫人，何等不美？郭芙脸上一红，说道：这许多英雄豪杰，谁不想当帮主？怎说得上‘安安稳稳’四字？耶律燕道：其实呢，也不用我哥哥上台。郭芙奇道：怎么？耶律燕道：刚才梁长老不是说么？当年丐帮大会君山，师母还不过十来岁，便以一条竹棒打得群雄束手归服，当上了帮主。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:1341
            "messages": [{"role": "user", "content": "姊弟三人奉父母之命，前赴晋阳邀请全真教耆宿长春子丘处机至襄阳主持英雄大会。这一日三姊弟从晋阳南归，却遭冰雪阻于风陵渡口，听了众人一番夜话。郭襄满脸喜色，低声自言自语：我生下来没到一天，他便抱过我了。转头对郭芙道：姊姊，那神雕侠小时候真在咱们桃花岛住过么？怎地我没听爹妈说起过？郭芙道：你知道什么？爹妈没跟你说的事多着呢。原来杨过断臂、小龙女中毒，全因郭芙行事莽撞而起。每当提及此事，郭靖便要大怒，女儿虽已出嫁，他仍要厉声呵责，不给女儿女婿留何情面，因此郭家大小对此事绝口不提，郭襄和郭破虏始终没听人说起过杨过之事。郭襄道：这么说来，他跟咱家很有交情啊，怎地一直没来往？嘿，九月十五襄阳城英雄大会，他定是要来与会的了。郭芙道：这人行事怪僻，性格儿又高傲得紧，他多半不会来。郭襄道：姊姊，咱们怎生想法儿送个请帖给他才好。转头向宋五道：宋五叔，你能想法子带个信给神雕侠么？宋五摇头道：神雕侠云游天下，行踪无定。他有事用得着兄弟们，便有话吩咐下来。我们要去找他，却一辈子也未必找得着。郭襄好生失望，她听各人说及杨过如何救王惟忠子裔、诛陈大方、审丁大全、赎宋五、杀人父而救人母种种豪侠义举，不由得悠然神往，听姊姊说自己幼时曾得他抱过，更加心中火热，恨不得能见他一面才好，待听说他多半不会来参与英雄大会，忍不住叹了口气，说道：英雄会上的人物不见得都是英雄，真正的大英雄大豪杰，却又未必肯去。突然间波的一声响，屋角中一人翻身站起，便是一直蜷缩成团、呼呼大睡那人。众人耳边厢但听得轰轰声响，原来是那人开口说话：姑娘要见神雕侠却也不难，今晚我领你去见他就是。众人听了那说话之声先已失惊，再看他形貌时，更大为诧异。但见他身长刚及四尺，躯体也甚瘦削，但大头、长臂、大手掌、大脚板，却又比平常人长大了许多，这副手脚和脑袋，便安在寻常人身上也已极不相称，他身子矮小，更显诡奇。郭襄大喜，说道：好啊，这位大叔，真正多谢了，我永远记得你的好心！只是我跟神雕侠素不相识，贸然求见，未免冒昧，又不知他见是不见。那矮子轰然道：你今日如不见他，只怕日后再也见不到了。郭襄道：只盼凭着前辈的金面，或许他肯见我。说时眉开眼笑，显得十分热切。郭芙站起身来，向那矮子道：请问尊驾高姓大名。那矮子冷笑道：天下似我这等丑陋之人，岂有第二人？你既不识，回去一问你爹爹妈妈便知。你父母为国为民，我素来十分敬仰，这个小妹妹爽快豪迈，又请我喝酒吃肉，我挺愿帮她个小忙。就在此时，远处缓缓传来一缕游丝般的声音，低声叫道：西山一窟鬼，十者到其九，大头鬼，大头鬼，此时不至，更待何时？这话声若断若续，有气无力，充满着森森鬼气，但一字一句，人人都听得明明白白。那大头矮子一怔，一声大喝，突然砰的一声响，火光一暗，那矮子已不知去向。众人齐吃一惊，见大门已然撞穿，原来那矮子竟破门跃出。撞破门板不奇，奇在一撞即穿，门板上给他撞破一个与他身形相似的大洞，此人跟着一撞之势从洞中跃出。郭破虏道：大姊，这矮子这等厉害！郭芙跟着父母，武林中人物见过不少，但这矮子却从未听父母说过，一时呆呆的说不出话来。郭襄却道：爹爹的授艺恩师江南七怪爷爷之中，便有一位矮个子的马王神韩爷爷。小弟，你乱叫人家矮子，爹爹知道了可要不依呢。你该称他一声前辈才是。郭靖对江南七怪的恩德一生念念不忘，推恩移爱，对任何盲人、矮子均礼敬有加，平素便如此教训子女。郭破虏尚未回答，忽听得呼的一声响，那大头矮子又已站在身前，北风夹雪，从破门中直吹进来，火堆中火星乱爆。郭芙怕那矮子出手伤了弟妹，抢上一步，挡在郭襄与郭破虏的身前。那矮子大头一摆，从郭芙腰旁探头过去，对郭襄道：小姑娘，你要见神雕侠，便同我去。郭襄道：好！大姊、小弟，咱们一块去罢。郭芙道：神雕侠有什么好见？你也别去。咱们和这位尊驾又素不相识。郭襄道：这位前辈大叔是好人！我去一会儿就回来，你们在这儿等我罢。宋五突然站起身来，说道：姑娘，千万去不得。这人是……是西山一窟鬼中的……中的人物，你去了……去了凶多吉少。那矮子咧嘴狞笑，说道：你知道西山一窟鬼？小姑娘说我是好人，你却说我们不是好人？左掌突然劈出，打在宋五肩头。砰的一声，宋五向后飞出，撞在墙上，登时晕去。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:710
            "messages": [{"role": "user", "content": "杨过安慰道：你爹爹新婚后心中高兴，定是待你更加好些。绿萼摇头道：我宁可他待我更凶些，也别娶新妈妈。杨过父母早死，对这般心情不大了然，有意要逗她开心，道：你新妈妈一定没你一半美。绿萼忙道：你偏说错了，我这新妈妈才真正是美人儿呢。爹爹可为她……为她……昨儿我们把那姓周的老头儿捉了来，若不是爹爹忙着安排婚事，决不会再让这老顽童逃走。杨过又惊又喜，问道：老顽童又逃走了？绿萼秀眉微蹙，道：可不是吗？杨过早料到以周伯通的本事，绝情谷中四弟子纵有渔网，也决拿他不住。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:151
            "messages": [{"role": "user", "content": "现下天下英雄会集于此，人人心怀忠义，咱们须得商量个妙策，使得蒙古鞑子不敢来犯我大宋江山。群雄纷纷起立，你一言我一语，都表赞同。此日来赴英雄宴之人多数都是血性汉子，眼见国事日非，大祸迫在眉睫，早就深自忧心，有人提起此事，忠义豪杰自是如响斯应。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },        
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:324
            "messages": [{"role": "user", "content": "黑衣僧一怔，觉得曾在什么地方和此人会过，又觉得他这眼色瞧得自己极不舒服，当即转头避开，过不片刻，忍不住又去望了他一眼。彭长老笑道：下得好大的雪啊，是不是？黑衣僧道：是，好大的雪。彭长老道：来，咱们去瞧瞧雪景。说着推开了板门。黑衣僧道：好，去瞧瞧雪景。站起身来，和他并肩站在门口。杨过虽隔着板壁，也觉彭长老眼光特异，心中隐隐有不祥之感。彭长老道：你师父说得好，杀人是万万不可的，但你全身劲力充溢，若不和人动手，心里便十分难过，是不是啊？黑衣僧迷迷糊糊的应道：是啊！彭长老道：你不妨发掌击这雪人，打好了，那可没有罪孽。黑衣僧望着雪人，双臂举起，跃跃欲试。这时离二僧到来之时已隔了小半个时辰，瘦丐身上又堆了一层白雪，连得他双眼也皆掩没。老道：你双掌齐发，打这雪人，打啊！打啊！打啊！语音柔和，充满了劝诱之意。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        }
    ]

    start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = len(data)) as executor: 
        future_to_data = {}
        
        for d in data:
            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": f"{''.join(random.choices('123456789', k=5))}",
            }
            future = executor.submit(fetch_post, url, headers, d)
            
            future_to_data[future] = d
            time.sleep(0.01) 
    for future in concurrent.futures.as_completed(future_to_data):
        result = future.result()
        results.append(result)

        print(f"Status: {result['status']}, Preview: {result['text']}")
        assert result['status'] == 200

    end_time = time.time()
    print(f"take: {end_time - start_time:.2f} seconds")
    log_file = f"{CUR_DIR}/nginx_access_balance.log"
    num_logs = 5
    print("\n=== verifying load balance ===")
    try:
        analyze_balance_basic(log_file, num_logs)
    
    except Exception as e:
        teardown_proxy_balance()
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")
    teardown_proxy_balance()

def analyze_balance_basic(log_file, num_logs=5):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"log {log_file} does not exist")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"failed to read log: {e}")
    
    if not logs:
        raise ValueError("empty log")
    parsed_logs = []
    for line in logs:
        try:
            data = parse_log_line(line) 
            if not data:
                continue
            parsed_logs.append({
                'prefill_idx': int(data['prefill_idx']),
                'decode_idx': int(data['decode_idx']),
                'promt_tks': int(data['promt_tks']),
                'decoded_tks': int(data['decoded_tks']),         
            })
        except KeyError as e:
            print(f"lack of description: {e} (origin log: {line[:100]}...)")

    if not parsed_logs:
        raise ValueError("could not find log")
    
    recent_logs = parsed_logs[-num_logs:]
    # list for network fluctuation
    prefill_counts = {0:0,1:0,2:0,3:0}
    decode_counts = {0:0,1:0,2:0,3:0}
    prompt_decode_min_idx = 3
    prompt_prefill_min_idx = 3
    for req in recent_logs:
        if req['promt_tks'] == 151:
            prompt_prefill_min_idx = req['prefill_idx']
            prompt_decode_min_idx = req['decode_idx']
        prefill_counts[req['prefill_idx']] += 1
        decode_counts[req['decode_idx']] += 1
    assert prefill_counts[prompt_prefill_min_idx] == 2 
    assert decode_counts[prompt_decode_min_idx] == 2 

def test_chat_completions_with_proxy_earliest(setup_teardown):

    proxy_port = find_free_port()
    ret = setup_proxy_earliest(proxy_port)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"  

    data = [
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length: 170
            "messages": [{"role": "user", "content": "他们约定在山下普光寺中聚会，以手击碑石为号。你无意中在碑上拍了一下，又显出功力惊人，无怪我那些没用的徒子徒孙便大惊小怪。那两个大魔头都是蒙古密教弟子，武功不弱，今年到中原几下出手，震动武林。你在桃花岛隐居，因而不知。那贵公子是蒙古的王子，据说还是大汗成吉思汗的近系子孙，旁人都叫他作霍都王子。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },                
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:595
            "messages": [{"role": "user", "content": "黄蓉道：回头我告知她便是，你爷儿俩去敌营走一趟，半天即回，又不是什么大事。杨过心想与黄蓉斗智，处处落于下风，但郭靖诚朴老实，决不是自己对手，同去蒙古军中后对付了他，再回来与小龙女会合不迟，于是略一结束，随同郭靖出城。郭靖骑的是汗血宝马，杨过乘了黄毛瘦马，两匹马脚力均快，不到半个时辰，已抵达蒙古大营。忽必烈听报郭靖竟然来到，又惊又喜，忙叫请进帐来。郭靖走进大帐，只见一位青年王爷居中而坐，方面大耳，两目深陷，不由得一怔：此人竟与他父亲拖雷一模一样。想起少年时与拖雷情深义重，此时却已阴阳相隔，不禁眼眶一红，险些儿掉下泪来。忽必烈下座相迎，一揖到地，说道：先王在日，时常言及郭靖叔叔英雄大义，小侄仰仰慕无已，日来得睹尊颜，实慰生平之愿。郭靖还了一揖，说道：拖雷安答和我情逾骨肉，我幼时母子俩托庇成吉思汗麾下，极仗令尊照拂。令尊英年，如日方中，不意忽尔谢世，令人思之神伤。说着不禁泪下。忽必烈见他言辞恳挚，动了真情，也不由得伤感，便与潇湘子、尹克西等一一引见，请郭靖上座。杨过侍立在郭靖身后，假装与诸人不识。国师等不知他此番随来是何用意，见他不理睬各人，也均不与他说话。麻光佐却大声道：杨兄……下面一个弟字还未出口，尹克西在他大腿上狠狠捏了一把。麻光佐啊哟一声，叫道：干什么？尹克西转过了头不理。麻光佐不知是谁捏他，口中唠唠叨叨骂人，便忘了与杨过招呼。郭靖坐下后饮了一杯马乳酒，不见武氏兄弟，正要动问，忽必烈已向左右吩咐：快请两位武爷。左右卫士应命而出，推了武敦儒、武修文进帐。两人手足都给用牛筋绳绑得结结实实，双足之间的牛筋长不逾尺，迈不开步子，只能慢慢的挨着过来。二武见到师父，满脸羞惭，叫了一声：师父！都低下了头不敢抬起。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:332
            "messages": [{"role": "user", "content": "洪七公听他这么说，连连点头，道：好小子，原来他是你义父。那知欧阳锋突然跃起，叫道：老叫化，咱们拳脚比不出胜败，再比兵器。洪七公听他叫自己老叫化，微微一笑，摇头道：不比啦，算你胜就是。欧阳锋道：什么算不算的？我非杀了你不可。回手折了根树枝，拉去枝叶，成为一条棍棒，向洪七公兜头击落。他的蛇杖当年纵横天下，厉害无比，现下杖头虽然无蛇，但这一杖击将下来，杖头未至，烈风已将杨过逼得难以喘气。杨过忙跃开躲避，看洪七公时，只见他拾起地下一根树枝，当作短棒，二人又已斗在一起。洪七公的打狗棒法世间无双，但轻易不肯施展，除此之外尚有不少精妙棒法，此时便逐一使将出来。这场拚斗，与适才比拚拳脚又是另一番光景，但见杖去灵蛇盘舞，棒来神龙夭矫，或似长虹经天，或若流星追月，只把杨过瞧得惊心动魄，如醉如痴。二人杖去棒来，直斗到傍晚，兀自难分胜败。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length: 980
            "messages": [{"role": "user", "content": "可是那彩雪蛛的毒性猛恶绝伦，他每一运气，胸口便烦恶欲呕，自顶至踵，每一处都麻痒难忍，不运气倒反而无事，连试三次都如此，废然叹道：唉，老顽童这一次可不好玩了！国师在外偷窥，却不知他有这等难处，暗想：不好，这老头儿在运内功了！心念一动，从怀中取出那只盛放彩雪蛛的金盒来，掀开盒盖，盒中十余只彩雪蛛蠕蠕而动，其时朝阳初升，照得盒中红绿斑斓，鲜艳夺目。国师从金盒旁的圆孔中拔出一根犀牛角做的夹子，夹起一根蛛丝，轻轻一甩，蛛丝上带着一只彩雪蛛，粘在山洞口左首。他连夹连甩，将盒中毒蛛尽数放出，每只毒蛛带着一根蛛丝，粘满了洞口四周。盒中毒蛛久未喂食，饥饿已久，登时东垂西挂，结起一张张蛛网，不到半个时辰，洞口已为十余张蛛网布满。当毒蛛结网之时，小龙女和周伯通看得有趣，均未出手干预，到得后来，一个直径丈余的洞口已满是蛛网，红红绿绿的毒蛛在蛛网上来往爬动，只瞧得心烦意乱。小龙女低声道：可惜我玉蜂针打完了，不然一针一个，省得这些毒蜘蛛在眼前爬来爬去的讨厌。周伯通拾起一枝枯枝，便想去揽蛛网，忽见一只大蝴蝶飞近洞口，登时给蛛网粘住。本来昆虫落入蛛网，定须挣扎良久，力大的还能毁网逃去，这只蝴蝶躯体虽大，一碰到蛛丝立即昏迷，动也不动。小龙女心细，叫道：别动，蛛丝有毒。周伯通吓了一跳，忙抛下枯枝。原来国师放毒蛛封洞，并非想以这些纤细的蛛网阻住二人，倒盼望他们出手毁网，游丝飞舞，免不了身上沾到一二根，剧毒便即入体。小龙女蓦地里想起，那日在古墓中教杨过轻功，杨过以天罗地网势捉到了一对白蝴蝶，当晚他做梦，梦到捉白蝴蝶，牢牢抓住了自己一对赤足，想着这些缱绻温馨的情景，不由得长长叹了口气，心中伤痛，珠泪双垂。周伯通观看毒蛛吃蝴蝶，大感兴趣，却觉得有点饿，又盘膝坐下，心想：反正我玄功一时不易恢复，多坐一会倒也不错。小龙女却想：这僵持之局不知何时方了？又不知道老顽童身上的毒性去尽没有？问道：你运功去毒，再有一天一晚可够了么？周伯通叹道：别说一天一晚，再有一百天一百晚也不管用。小龙女惊道：那怎生是好？周伯通笑道：那贼秃若肯送饭给咱们吃，在这山洞中住上几年，也没什么不好。小龙女道：他不肯送饭的。叹了口气，道：倘若杨过在这儿，我便在这山洞中住一辈子也没什么。周伯通怒道：我什么地方及不上杨过了？他还能比我强么？我陪着你又有什么不好？他这两句话不伦不类，小龙女却也不以为忤，只淡淡一笑，道：杨过会使全真剑法，我和他双剑合璧，便能将这和尚杀得落荒而逃。周伯通道：哼，全真剑法有什么了不起？我是全真派大长老，我难道不会使？杨过能胜得我么？小龙女道：我们这双剑合璧，叫作玉女素心剑法，要我心中爱他，他心中爱我，两心相通，方能克敌制胜。周伯通一听到男女之爱，立时心惊肉跳，连连摇手，说道：休提，休提。我不来爱你，你也千万别来爱我。我跟你说，在山洞中住了几年也没什么大不了。当年我在桃花岛山洞中孤零零的住了十多年，没人相伴，只得自己跟自己打架，现今跟你在一起，有说有笑，那就大不相同了。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:1084
            "messages": [{"role": "user", "content": "国师道：这张唐卡上绣的祖师爷，是莲华生大士，我们一齐向祖师爷礼拜。郭襄便随着国师向画像礼拜致敬。国师道：祖师爷右手拿的是文殊菩萨的智能之剑，把各种各样乱七八糟的烦恼妄想全部斩断。他左手这朵莲花，是教人心里清净平和，就像莲花一样，没半点污秽渣滓，只有澄澈露水，美丽安静。郭襄见绣像中的莲华生大士慈悲庄严，登时肃然起敬。国师又道：我从今天起，教你修报身佛金刚萨埵所说的瑜珈密乘，修成之后，再修法身佛普贤菩萨所说的大瑜珈密乘、无比瑜珈密乘，一直到最后的无上瑜珈密乘。郭襄问道：师父，要修成无上瑜珈密乘，那得多少时候啊？国师道：无上瑜珈密乘无穷无尽，永远说不上修成，也说不上要多少时候。郭襄道：那你也没修成了？国师叹了口气，道：是啊，倘若我修得稍有成就，怎么还会去苦练那龙象般若功？还会起心来和杨过、小龙女决一胜败？真是蠢才！郭襄道：谁说你蠢了？不决一胜败，又怎知谁蠢谁聪明？国师又长长叹了口气，说道：我先教你六字大明咒：唵、嘛、呢、叭、咪、吽，你诚心诚意跟我念一遍。郭襄学着念了，口音略有不准，国师给她纠正。郭襄道：师父，祖师爷是好人，我早晚拜他，不过我不学驱除烦恼的法门。国师问道：为什么不学？郭襄道：我喜欢心里有烦恼！心道：没了烦恼，就没了大哥哥，我喜欢心里有大哥哥！国师口念密宗真言，盼求上师慈悲加持，感化郭襄发心去修学瑜珈密乘。他这一派的教法，讲究缘法以及修习者的诚意发愿，外人不得勉强，他那知郭襄这时心中想的却是：可惜我迟生了二十年。倘若妈妈先生我，再生姊姊，我学会了师父的龙象般若功和无上瑜珈密乘，在全真教道观外住了下来，自称大龙女，小杨过在全真教中受师父欺侮，逃到我家里，我收留了他教他武功，他慢慢的自会跟我好了。他再遇到小龙女，最多不过拉住她手，给她三枚金针，说道：‘小妹子，你很可爱，我心里也挺喜欢你。不过我的心已属大龙女了。请你莫怪！你有什么事，拿一枚金针来，我一定给你办到。’唉，还有一枚金针，我要请他不管发生了什么事，无论如何不可自尽。他是扬名天下的神雕大侠，千金一诺，不，万金一诺，万万金一诺，答允了我的话不可不守信约，不能自尽就一生一世决不能自尽。天时渐寒，郭襄一算日子，杨过与小龙女十六年之约将届，从荆湖南路缓缓而去绝情谷，差不多也要一个月时候，说道：师父，你到底敢不敢去跟杨过、小龙女比武？你一个人打不过，我们师徒二人联手，使几招无上瑜珈密乘好了。金轮国师哈哈一笑，说道：好！咱俩明天启程，去绝情谷会会‘玉女素心剑法’！他与郭襄相处既久，对她甚为喜爱，早已改变初衷，不再想将她折磨，胁迫郭靖降顺。国师和郭襄起行赴绝情谷时，杨过已早了一日启程。三人相距不过百余里而已。郭靖与黄蓉自幼女出走，日夕挂怀。其后派出去四处打探的丐帮弟子一一回报，均说不知音讯。又过十余日，突然程英和陆无双到了襄阳，传来柯镇恶的讯息，说道郭襄已遭掳入蒙古军中。郭靖、黄蓉大惊。当晚黄蓉便和程英两人暗入蒙古军营，四下查访，也如杨过一般，在北大营探不到丝毫端倪。第三晚更和蒙古众武士斗了一场，四十余名武士将黄蓉和程英团团围住，总算黄程两人武功了得，黄蓉又连使诡计，这才闯出敌营，回归襄阳。黄蓉心下计议，瞧情势女儿并非在蒙古军中，但迄今得不到半点音讯，决非好兆，探得蒙古大军又在征集粮草，并无即行南攻的迹象，与郭靖商议了，便即出城寻访。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },        
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length:1064
            "messages": [{"role": "user", "content": "杨过与小龙女互视一眼，均想：我二人若能撇开了旁人，在静室中相处片刻，死亦甘心。当即携手向西，从侧门出去，走过两间房，来到第三间房前。小龙女眼光始终没离开杨过之脸，见房门闭着，也不细看，伸手推开，正要跨过门槛进去，杨过猛地想到一事，忙伸手拉住道：小心了。小龙女道：怎么？杨过左足踏在门槛之外，右足跨过门槛往地板上一点，立即缩回，丝毫不见异状。小龙女道：你怕谷主要暗害咱们吗？他这人很好，决不致于……刚说完这三句话，猛听得嗤嗤声响，眼前白光闪动，八柄利剑自房门上下左右挺出，纵横交错，布满入口，若有人于此时踏步进门，武功再高，也难免给这八柄利剑从四面八方在身上对穿而过。小龙女透了口长气，说道：过儿，这谷主恁地歹毒，我真瞧错他的为人了。咱们也不用跟他比什么剑，这就走罢。忽听身后有人说道：谷主请两位入室拣剑。两人回过头来，见八名绿衫弟子手持带刀渔网，拦在身后，自是谷主防杨龙二人相偕逃走，派人截住后路。小龙女的金铃索已为黑剑割断，再不能如适才这般遥点绿衫弟子的穴道。小龙女向杨过道：你说这室中还有什么古怪？杨过将她双手握在掌中，说道：姑姑，此刻你我相聚，复有何憾？便万剑穿心，你我也死在一起。小龙女心中也是柔情万种。两人一齐步入剑室，杨过随手把门带上。只见室中壁上、桌上、架上、柜中、几间，尽皆列满兵刃，式样繁多，十之八九都是古剑，或长逾七尺，或短仅数寸，有的铁锈斑驳，有的寒光逼人，二人眼光缭乱，一时也看不清这许多。小龙女对杨过凝视半晌，突然嘤的一声，投入他怀中。杨过将她紧紧抱住，在她嘴上亲去。小龙女在他一吻之下，心魂俱醉，双手伸出去搂住他头颈，凑嘴回吻。突然砰的一声，室门推开，一名绿衫弟子厉声说道：谷主有令，拣剑后立即出室，不得逗留。杨过脸上一红，当即双手放开。小龙女却想自己心爱杨过，二人相拥而吻决没什么不该，但既有人在旁干扰，难以畅怀，叹了一口气，轻声说道：过儿，待咱们打败了那谷主，你再这般亲我。杨过笑着点了点头，伸左手搂住她腰，柔声道：我永生永世也亲你不够。你拣兵器罢。小龙女道：这里的兵刃瞧来果然均是异物，没一件不好。咱们古墓里也没这么多。于是先从壁间逐一看去，要想拣一对长短轻重都是一般的利剑，但瞧来瞧去，各剑均自不同。她一面看，一面问道：适才进室之时，你怎知此处装有机关？杨过道：我从谷主的脸色和眼光中猜想而知。他本想娶你为妻，但听到你要和我联手斗他，便想杀你了。以他为人，我不信他会好心让咱们来拣选兵刃。小龙女又低低叹了口气，道：咱们使玉女素心剑法，能胜得了他么？杨过道：他武功虽强，却也并不在金轮国师之上。我二人联手胜得国师，谅来也可胜他。小龙女道：是了，国师不住激他和我二人动手，他是要瞧个清楚。杨过微笑道：人心鬼蜮，你也领会得一些了。我只担心你身子，刚才你又呕了血。小龙女笑靥如花，道：你知道的，我伤心气恼的时候才会呕血，现下我欢喜得很，这点内伤不算什么。你也呕了血，不打紧罢？杨过道：我见了你，什么都不碍事了。小龙女柔声道：我也这样。顿了一顿，又道：你近来武功大有进境，合斗国师之时咱们尚且能胜，何况今日？杨过听了此言，也觉这场比试定能取胜，握着她手说道：我想要你答允一件事，不知你肯不肯？小龙女柔声道：你又何必问我？我早已不是你师父，是你妻子啦。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        },
        {
            "model": "deepseek",
            "temperature": 0,
            "max_tokens": 20,#prompt length: 1398
            "messages": [{"role": "user", "content": "听那湖北客人续道：襄阳城中数十万军民也人人竭力死守，没一个畏缩退后的。像小人只是个推车的小商贩，也搬土运石，出了一身力气来帮助守城。我脸上这老大箭疤，便是给蒙古鞑子射的。众人一齐望他脸上，见他左眼下果然有个茶杯口大小的箭创，不由得都肃然起敬。那广东客人道：我大宋土广人多，倘若人人都像老兄一样，蒙古鞑子再凶狠十倍，也不能占我江山。那湖北人道：是啊，你瞧蒙古大军连攻襄阳十余年，始终打不下，别的地方却是手到拿来。听说西域域外几十个国家都给蒙古兵灭了，我们襄阳始终屹立如山。蒙古王子忽必烈亲临城下督战，可也奈何不了我们襄阳人。说着大有得意之色。那广东客人道：老百姓都是要跟鞑子拼命的，鞑子倘若打到广东来，我们广东佬也好好跟他妈的干一下子。那湖北人道：不跟鞑子拼命，一般的没命。蒙古鞑子攻不进襄阳，便捉了城外的汉人，绑在城下一个个的斩首，还把四五岁、六七岁的小孩儿用绳子绑了，让马匹拉着，拖到城下绕城奔跑，绕不到半个圈子，孩子早没了气。我们在城头听到孩儿们啼哭呼号，真如刀割心头一般。鞑子只道使出这等残暴手段，便能吓得我们投降，可是他越狠毒，我们越守得牢。那一年襄阳城中粮食吃光了，水也没得喝了，到后来连树皮污水也吃喝干净，鞑子却始终攻不进来。后来鞑子没法子，只有退兵。那广东人道：这十多年来，若不是襄阳坚守不屈，咱们大宋半壁江山，只怕早不在了。众人纷纷问起襄阳守城的情形，那湖北人说得有声有色，把郭靖、黄蓉夫妇夸得便如天神一般，众人赞声不绝。一个四川口音的客人忽然叹道：其实守城的好官勇将各地都有，就只朝廷忠奸不分，往往奸臣享尽荣华富贵，忠臣却含冤而死。前朝的岳爷爷不必说了，比如我们四川，朝廷就屈杀了好几位守土的大忠臣。那湖北人道：那是谁啊？倒要请教。那四川人道：蒙古鞑子攻打四川十多年，全赖余玠余大帅守御，全川百姓都当他万家生佛一般。那知皇上听信了奸臣丁大全的话，说余大帅什么擅权，又是什么跋扈，赐下药酒，逼得他自杀了，换了一个懦弱无能的奸党来做元帅。后来鞑子一攻，川北当场便守不住。阵前兵将是余大帅的旧部，大家一样拼命死战。但那元帅只会奉承上司，一到打仗，调兵遣将什么都不在行，自然抵挡不住了。丁大全、陈大方这伙奸党庇护那狗屁元帅，反冤枉力战有功的王惟忠将军通敌，竟将他全家逮京，把王将军斩首了。他说到这里，声音竟有些呜咽，众人同声叹息。那广东客人愤愤的道：国家大事，便坏在这些奸臣手里。听说朝中三犬，这奸臣丁大全便是其中一犬了。一个白净面皮的少年一直在旁听着，默不作声，这时插口道：不错，朝中奸臣以丁大全、陈大方、胡大昌三人居首。临安人给他们名字那个‘大’字之旁都加上一点，称之为丁犬全、陈犬方，胡犬昌。众人听到这里都笑了起来。那四川人道：听老弟口音，是京都临安人氏了。那少年道：正是。那四川人道：然则王惟忠将军受刑时的情状，老弟可曾听人说起过？那少年道：小弟还亲眼看见呢。王将军临死时脸色兀自不变，威风凛凛，骂丁大全和陈大方祸国殃民，而且还有一件异事。众人齐问：什么异事？那少年道：王将军是陈大方一手谋害的。王将军被绑赴刑场之时，在长街上高声大叫，说死后决向玉皇大帝诉冤。王将军死后第三天，那陈大方果在家中暴毙，他的首级却高悬在临安东门的钟楼檐角之上，在一根长竿上高高挑着。这地方猿猴也爬不上去，别说是人了，若不是玉皇大帝派的天神天将，却是谁干的呢？众人啧啧称奇。那少年道：此事临安无人不晓，却非我生安白造的。各位若到临安去，一问便知。那四川人道：这位老弟的话的确不错。只不过杀陈大方的，并不是天神天将，却是一位英雄豪杰。那少年摇头道：想那陈大方是朝中大官，家将亲兵，防卫何等周密，常人怎杀得了他？再说，要把这奸臣的首级高高挑在钟楼的檐角之上，除非是生了翅膀，才有这等本领。那四川人道：本领非凡的奇人侠士，世上毕竟还是有的。但小弟若不是亲眼目睹，可也真的难以相信。那少年奇道：你亲眼见他把陈大方的首级挂上高竿？你怎会亲眼看见？那四川人微一迟疑，说道：王惟忠将军有个儿子，王将军遭逮时他逃走在外。朝中奸臣要斩草除根，派下军马追拿。那王将军之子也是个军官，虽会武艺，却寡不敌众，眼见要便给抓住，却来了一位救星，赤手空拳的将数十名军马打得落花流水。小王将军便将父子卫国力战、却让奸臣陷害之情说了。那位大侠连夜赶赴临安，想要搭救王将军，但终于迟了两日，王将军已经遭害。请基于以上故事情节续写小说，字数不少于1200字。"}],
            "stream": True
        }
          
    ]
    
    start_time = time.time()
    results = []
    POST_DATA_COUNT = 25

    with concurrent.futures.ThreadPoolExecutor(max_workers = len(data)) as executor: 
        future_to_data = {}
        
        for d_index in range(POST_DATA_COUNT):
            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": f"{''.join(random.choices('123456789', k=5))}",
            }
            future = executor.submit(fetch_post, url, headers, data[d_index % len(data)])
            future_to_data[future] = data[d_index % len(data)]
            time.sleep(0.01) 
    
    for future in concurrent.futures.as_completed(future_to_data):
        result = future.result()
        results.append(result)

        print(f"Status: {result['status']}, Preview: {result['text']}")
        assert result['status'] == 200

    end_time = time.time()
    print(f"take: {end_time - start_time:.2f} seconds")
    log_file = f"{CUR_DIR}/nginx_access_balance.log"
    num_logs = POST_DATA_COUNT
    print("\n=== verifying load balance ===")
    try:
        analyze_balance_earliest(log_file, num_logs, len(data))
    
    except Exception as e:
        teardown_proxy_balance()
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")
    teardown_proxy_balance()

def analyze_balance_earliest(log_file, num_logs, num_data):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"log {log_file} does not exist")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"failed to read log: {e}")
    
    if not logs:
        raise ValueError("empty log")
    parsed_logs = []
    for line in logs:
        try:
            data = parse_log_line(line) 
            if not data:
                continue
            parsed_logs.append({
                'prefill_idx': int(data['prefill_idx']),
                'decode_idx': int(data['decode_idx']),
                'promt_tks': int(data['promt_tks']),
                'decoded_tks': int(data['decoded_tks']),
                'rcved': float(data['rcved'])
            })
        except KeyError as e:
            print(f"lack of description: {e} (origin log: {line[:100]}...)")

    if not parsed_logs:
        raise ValueError("could not find log")
    
    recent_logs = parsed_logs[-num_logs:]
    recent_logs.sort(key=lambda x: x['rcved'])
    # pick static mapping for earliest algo
    recent_logs = recent_logs[: num_data]
    '''
    [NOTICE] idx depends on PREFILL_NUM
             probably report wrong for network fluctuation or any optimization on processing prompt length in future

    '''
    prefill_mapping = {170:0, 332:[1,2], 595:[1,2], 980:3, 1084:0, 1064:1, 1398: [0, 2]}
    decode_counts = {0:0, 1:0, 2:0, 3:0}
    prompt_decode_min_idx = 3
    for req in recent_logs:
        val = prefill_mapping[req['promt_tks']]
        if isinstance(val, int):
            assert req['prefill_idx'] == val
        else:  # list
            assert req['prefill_idx'] in val
        if req['promt_tks'] == 170:
            prompt_decode_min_idx = req['decode_idx']
        decode_counts[req['decode_idx']] += 1

    assert decode_counts[prompt_decode_min_idx] == 2 

def test_chat_completions_with_proxy_concurrent(setup_teardown):
    proxy_port = find_free_port()
    ret = setup_proxy_basic(proxy_port)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions" 

    start_time = time.time()
    results = []
    POST_DATA_COUNT = 2000

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(PREFILL_NUM, 200)) as executor:
        futures = []
        
        for i in range(POST_DATA_COUNT):

            data = [
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Glacierhaven Village is built near a large glacier, and the villagers can control ice and snow—some use it to make ice sculptures, others use it to store food, and a few can even create ice bridges to cross rivers. Ava, an 11-year-old girl in the village, cannot control any ice or snow. No matter how she tries to freeze water, it remains liquid. The other children tease her, calling her 'the Iceless Girl' and not letting her play in the snow. Ava often sits by the glacier, watching the ice sparkle in the sun and feeling sad. One afternoon, a penguin with black and white feathers waddles up to her. The penguin is the guardian of the glacier, and it tells Ava that her power is not controlling ice and snow, but communicating with the glacier's creatures—penguins, seals, arctic hares, and even the ice spirits. Ava is doubtful at first. The penguin asks her to put her hand on the glacier. When she does, she suddenly hears a series of voices—the seals are telling her that the glacier is melting faster than usual, and the ice spirits are warning that the melting ice will cause a flood soon. Ava's heart fills with excitement. She has finally found her own power. Please continue this story, describing Ava's process of learning to use her power to communicate with glacier creatures, the help she provides to the villagers (such as warning of the melting glacier, preparing for the flood), the changes in the villagers' attitude towards her, the danger the village faces (such as a massive iceberg breaking off and causing a huge flood that threatens the village), and how Ava uses her power to guide the villagers to safety. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Honeyglow Village is located near a large beehive, and the villagers can communicate with bees and collect honey easily—some use bees to pollinate crops, others use honey to make food, and a few can even use bee venom to heal certain diseases. Lucas, a 10-year-old boy in the village, is afraid of bees, and they never respond to his calls. Whenever he approaches the beehive, the bees buzz around him angrily. The other children laugh at him, calling him 'the Bee-Fearer' and excluding him from honey-collecting activities. Lucas often stands far away from the beehive, watching the others work with bees and feeling lonely. One morning, a queen bee with golden stripes flies to him and hovers in front of his face. The queen bee is the leader of the beehive, and it tells Lucas that his fear of bees is a sign of his special power—he can communicate with the plants that the bees rely on, and he can ensure that the plants bloom and provide enough nectar for the bees. Lucas is skeptical at first. The queen bee asks him to touch a nearby clover. When he does, he suddenly hears the clover's voice—it is telling him that it needs more sunlight and water to bloom. He also hears the other plants in the area whispering that a drought is coming, which will make them unable to produce nectar. Lucas is overjoyed. He has found his own power. Please continue this story, describing Lucas's process of learning to use his power to communicate with plants, the help he gives to the villagers (such as helping the plants grow, ensuring a steady supply of honey), the changes in the villagers' attitude towards him, the danger the village faces (such as the drought causing the bees to leave and the crops to fail), and how Lucas uses his power to help the plants survive the drought and keep the bees in the village. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Emberwood Village is built in a forest where the trees have glowing embers on their branches, and the villagers can collect these embers and use them for light and heat—some use them to light their houses, others use them to cook, and a few can even use them to ward off cold spirits. Maya, an 11-year-old girl in the village, cannot collect any embers. Whenever she tries to touch an ember, it goes out immediately. The other children tease her, calling her 'the Ember-Extinguisher' and not letting her near the glowing trees. Maya often sits under a non-glowing tree, watching the embers flicker and feeling sad. One evening, a firefly with glowing red wings flies to her. The firefly is the guardian of the ember trees, and it tells Maya that her power is not collecting embers, but communicating with the cold spirits in the forest and calming them down. It says that the embers go out around her because her power balances the heat of the embers, making the cold spirits less aggressive. Maya is doubtful at first. The firefly asks her to walk into the deeper part of the forest where the cold spirits are most active. When she does, she suddenly hears faint whimpering sounds—the cold spirits are telling her that their home is being destroyed by the excessive heat of the embers, and they are forced to attack the village. Maya's eyes light up. She has finally found her own power. Please continue this story, describing Maya's process of learning to use her power to communicate with cold spirits, the help she provides to the villagers (such as calming the cold spirits, preventing them from attacking the village), the changes in the villagers' attitude towards her, the danger the village faces (such as a group of enraged cold spirits launching a large-scale attack to put out all the embers), and how Maya uses her power to mediate between the villagers and the cold spirits and save the village. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Coralreef Village is built on a small island surrounded by coral reefs, and the villagers can control coral and use it to build houses and defend the island—some use coral to make strong walls, others use it to trap fish for food, and a few can even make coral grow to block incoming ships. Jack, a 12-year-old boy in the village, cannot control any coral. No matter how he touches the coral reefs, they remain unchanged. The other children mock him, calling him 'the Coralless Boy' and excluding him from island defense activities. Jack often sits on the beach, watching the coral reefs and feeling lonely. One morning, a seahorse with purple scales swims to the shore and stops in front of him. The seahorse is the guardian of the coral reefs, and it tells Jack that his power is not controlling coral, but communicating with the creatures living in the coral reefs—seahorses, clownfish, crabs, and even the coral spirits. Jack doesn't believe it at first. The seahorse asks him to put his hand into the water near the coral reef. When he does, he suddenly hears a series of voices—the clownfish are telling him that a large ship is about to sail into the coral reefs and destroy them, and the coral spirits are warning that the destruction of the reefs will leave the island unprotected from storms. Jack is excited. He has found his own power. Please continue this story, describing Jack's learning to use his power to communicate with coral reef creatures, the help he gives to the villagers (such as warning them of the incoming ship, protecting the coral reefs), the changes in the villagers' attitude towards him, the danger the village faces (such as a hurricane approaching the island after the coral reefs are damaged), and how Jack uses his power to guide the villagers to repair the coral reefs and protect the island from the hurricane. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Duststorm Village is located in a desert oasis, and the villagers can control sand and use it to protect the oasis—some use sand to make barriers around the oasis, others use it to filter water, and a few can even create sandstorms to drive away desert bandits. Emma, a 10-year-old girl in the village, cannot control any sand. Whenever she tries to pick up sand, it slips through her fingers. The other children tease her, calling her 'the Sandless Girl' and not letting her play in the sand. Emma often sits by the oasis's spring, watching the sand dunes and feeling sad. One afternoon, a desert fox with golden fur walks up to her. The fox is the guardian of the oasis, and it tells Emma that her power is not controlling sand, but communicating with the desert's water spirits and finding hidden water sources. It says that her inability to hold sand is because her power is connected to water, which is the opposite of sand. Emma is skeptical at first. The fox asks her to close her eyes and feel the ground under her feet. When she does, she suddenly feels a faint vibration—the water spirits are telling her that the oasis's spring is drying up, and there is a hidden underground river not far from the village. Emma's heart fills with excitement. She has finally found her own special power. Please continue this story, describing Emma's process of learning to use her power to communicate with water spirits, the help she provides to the villagers (such as finding the hidden underground river, saving the oasis's spring), the changes in the villagers' attitude towards her, the danger the village faces (such as a severe sandstorm combined with the drying up of the spring, threatening to make the oasis disappear), and how Emma uses her power to guide the villagers to tap the underground river and save the oasis. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Starfall Village is located on a mountain where shooting stars often fall, and the villagers can collect stardust from the shooting stars and use it to enhance their abilities—some use it to make their strength stronger, others use it to improve their speed, and a few can even use it to heal serious injuries. Liam, an 11-year-old boy in the village, cannot collect any stardust. Whenever a shooting star falls, the stardust passes right through his hands. The other children laugh at him, calling him 'the Stardustless Boy' and excluding him from stardust-collecting activities. Liam often sits on the mountain top, watching the shooting stars and feeling lonely. One night, a shooting star falls near him and turns into a small glowing creature. The creature is a star spirit, and it tells Liam that his power is not collecting stardust, but communicating with the stars and understanding their predictions—he can tell the village's future by observing the stars' positions and movements. Liam doesn't believe it at first. The star spirit asks him to look up at the stars. When he does, he suddenly understands the pattern of the stars—they are predicting that a meteor shower will hit the village soon, and there is also a sign that a rare medicinal herb that can cure the village elder's illness is growing on the other side of the mountain. Liam is overjoyed. He has found his own power. Please continue this story, describing Liam's learning to use his power to communicate with the stars, the help he gives to the villagers (such as finding the medicinal herb, warning of the meteor shower), the changes in the villagers' attitude towards him, the danger the village faces (such as the meteor shower causing fires and destroying houses), and how Liam uses his power to guide the villagers to avoid the meteor shower's impact and put out the fires. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Mushroomglade Village is built in a forest full of giant mushrooms, and the villagers can control the growth of mushrooms—some use them to build houses, others use them to make food, and a few can even use mushroom spores to put enemies to sleep. Zoe, a 10-year-old girl in the village, cannot make any mushrooms grow. No matter how she cares for the mushroom spores, they never germinate. The other children tease her, calling her 'the Mushroomless Girl' and not letting her play in the mushroom forest. Zoe often sits under a giant mushroom cap, watching the others tend to the mushrooms and feeling sad. One morning, a snail with a shell covered in mushroom patterns crawls onto her hand. The snail is the guardian of the mushroom forest, and it tells Zoe that her power is not controlling mushrooms, but communicating with the fungi and bacteria in the soil that help mushrooms grow. She can tell when the soil is lacking nutrients and help improve it to promote mushroom growth. Zoe is doubtful at first. The snail asks her to touch the soil under a giant mushroom. When she does, she suddenly hears a series of tiny voices—the fungi are telling her that the soil is lacking nitrogen, and the bacteria are warning that a harmful mold is spreading in the soil, which will kill the mushrooms. Zoe's eyes light up. She has finally found her own power. Please continue this story, describing Zoe's process of learning to use her power to communicate with soil fungi and bacteria, the help she provides to the villagers (such as improving the soil, eliminating the harmful mold), the changes in the villagers' attitude towards her, the danger the village faces (such as the harmful mold spreading rapidly and threatening to destroy all the mushrooms in the forest), and how Zoe uses her power to stop the mold and save the mushroom forest and the village. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Waterfall Village is built beside a large waterfall, and the villagers can control the flow of the waterfall and use its energy—some use it to turn watermills, others use it to generate power, and a few can even jump through the waterfall to reach a hidden cave. Ethan, a 12-year-old boy in the village, cannot control the waterfall's flow. Whenever he tries to touch the waterfall, the water pushes him away. The other children mock him, calling him 'the Waterfall-Rejected' and excluding him from activities near the waterfall. Ethan often sits on a rock beside the waterfall, listening to the water's roar and feeling lonely. One afternoon, a water snake with silver scales swims up to the shore and stops in front of him. The snake is the guardian of the waterfall, and it tells Ethan that his power is not controlling the waterfall's flow, but understanding the water's memory—he can see the images of what has happened in the waterfall's waters over the years. Ethan doesn't believe it at first. The snake asks him to stare at the waterfall's water. When he does, he suddenly sees images in the water—the village being built beside the waterfall, a group of travelers getting lost and being saved by the villagers, and a hidden treasure hidden behind the waterfall by the village's ancestors. He also sees a warning in the water's memory: the waterfall's rocks are loosening, and it will collapse soon. Ethan is excited. He has found his own power. Please continue this story, describing Ethan's learning to use his power to see the water's memory, the help he gives to the villagers (such as finding the hidden treasure, warning of the waterfall's collapse), the changes in the villagers' attitude towards him, the danger the village faces (such as the waterfall collapsing and causing a flood that threatens to destroy the village), and how Ethan uses his power to guide the villagers to reinforce the waterfall's rocks and evacuate to a safe place. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Sunflower Village is located in a field full of sunflowers that always face the sun, and the villagers can use the sunflowers' energy to enhance their mood and strength—some use it to stay happy even in difficult times, others use it to work longer hours, and a few can even use it to heal emotional wounds. Mia, an 11-year-old girl in the village, cannot feel any energy from the sunflowers. No matter how she stands among them, she remains sad and tired. The other children tease her, calling her 'the Sunflowerless Girl' and not letting her play in the sunflower field. Mia often sits at the edge of the field, watching the sunflowers turn towards the sun and feeling lonely. One morning, a ladybug with orange spots crawls onto her shoulder. The ladybug is the guardian of the sunflower field, and it tells Mia that her power is not absorbing the sunflowers' energy, but communicating with the sun's rays and guiding the sunflowers to grow better. She can tell when the sunflowers need more sunlight and help them adjust their direction to absorb more energy. Mia is skeptical at first. The ladybug asks her to touch a sunflower that is wilting. When she does, she suddenly hears the sunflower's voice—it is telling her that it cannot get enough sunlight because it is blocked by a tall tree. She also hears the other sunflowers whispering that a group of birds is going to eat their seeds soon. Mia's heart fills with excitement. She has finally found her own special power. Please continue this story, describing Mia's process of learning to use her power to communicate with the sun's rays and sunflowers, the help she provides to the villagers (such as moving the blocking tree, driving away the birds), the changes in the villagers' attitude towards her, the danger the village faces (such as a long period of cloudy weather that makes the sunflowers wilt, depriving the villagers of their energy source), and how Mia uses her power to guide the sunflowers to absorb the little sunlight available and help them survive until the sun comes out again. The word count should be no less than 1200 words."}], 
                    "stream": True
                },
                {
                    "model": "deepseek", 
                    "temperature": 0, 
                    "max_tokens": 20, 
                    "messages": [{"role": "user", "content": "Cave dweller Village is built inside a large cave, and the villagers can see in the dark and control the glow of cave crystals—some use it to find their way in the cave, others use it to light up the village, and a few can even use the crystals to detect hidden passages. Ryan, a 10-year-old boy in the village, cannot see in the dark and cannot control the cave crystals. Whenever he enters the dark part of the cave, he has to feel his way around, and the crystals never glow for him. The other children laugh at him, calling him 'the Cave Blind Boy' and excluding him from cave exploration activities. Ryan often stays in the lit part of the cave, watching the others explore the dark passages and feeling sad. One evening, a bat with large ears flies to him and perches on his hand. The bat is the guardian of the cave, and it tells Ryan that his power is not seeing in the dark or controlling crystals, but communicating with the cave's creatures—bats, spiders, mice, and even the cave spirits. He can use their eyes and ears to 'see' and 'hear' in the dark. Ryan doesn't believe it at first. The bat asks him to close his eyes and let the cave's creatures guide him. When he does, he suddenly feels as if he can see the entire cave through the bats' eyes—he sees a hidden passage that leads to a underground spring, and he also sees a group of poisonous snakes approaching the village from the dark part of the cave. Ryan is overjoyed. He has found his own power. Please continue this story, describing Ryan's learning to use his power to communicate with cave creatures, the help he gives to the villagers (such as finding the underground spring, warning of the poisonous snakes), the changes in the villagers' attitude towards him, the danger the village faces (such as the cave's ceiling collapsing in the dark part, threatening to block the village's exit), and how Ryan uses his power to guide the villagers to find a new exit and avoid the collapse. The word count should be no less than 1200 words."}], 
                    "stream": True
                }
            ]

            headers = {
                "Content-Type": "application/json",
                "X-Request-Id": ''.join(random.choices('123456789', k=5)),
            }

            future = executor.submit(fetch_post, url, headers, data[i % len(data)])
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                # print(f"Status: {result['status']}, Preview: {result.get('text', 'Error')}")
                # assert result.get('status') == 200
            except Exception as e:
                print(f"Error: {e}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    log_file = f"{CUR_DIR}/nginx_access_balance.log"
    num_logs = POST_DATA_COUNT
    print("\n=== verifying load balance ===")
    try:
        analysis_result = analyze_balance_concurrent(log_file, num_logs)
        for idx in sorted(analysis_result['prefill_frequency'].keys()):
            count = analysis_result['prefill_frequency'][idx]
            assert num_logs // PREFILL_NUM - 30 <= count <= num_logs // PREFILL_NUM + 30

        for idx in sorted(analysis_result['decode_frequency'].keys()):
            count = analysis_result['decode_frequency'][idx]
            assert num_logs // DECODE_NUM - 30 <= count <= num_logs // DECODE_NUM + 30

    except Exception as e:
        teardown_proxy_balance()
        print(f"\n=== verifying fail: {e} ===")
        raise
    print("\n=== verifying pass ===")
    teardown_proxy_balance()

def analyze_balance_concurrent(log_file, num_logs):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"log {log_file} does not exist")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"failed to read log: {e}")
    
    if not logs:
        raise ValueError("empty log")
    
    prefill_freq = defaultdict(int)
    decode_freq = defaultdict(int)
    
    recent_logs = logs[-num_logs:] 
    
    for line in recent_logs:
        try:
            data = parse_log_line(line)
            if not data:
                continue
                
            prefill_val = data.get('prefill_idx')
            decode_val = data.get('decode_idx')
            
            if prefill_val is not None:
                prefill_freq[int(prefill_val)] += 1
            if decode_val is not None:
                decode_freq[int(decode_val)] += 1
        except Exception as e:
            print(f"Error parsing line: {e} (log: {line[:100]}...)")
            continue

    return {
        'prefill_frequency': dict(prefill_freq),
        'decode_frequency': dict(decode_freq)
    }

def parse_log_line(line):

    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None  

    line = line[1:-1]  
    parts = []
    current = ""
    in_string = False 

    for char in line:
        if char == '"' and not in_string:
            in_string = True 
        elif char == '"' and in_string:
            in_string = False 
        elif char == "," and not in_string:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:
        parts.append(current)

    result = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        result[key] = value

    return result


