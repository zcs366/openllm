#!/usr/bin/env python3
"""
Tr（跨会话记忆迁移率）测量实验 ═══
Tr = Session A存入记忆中被Session B检索命中的比例
失败分类：retrieval_noise / injection_skip / context_pressure / model_confusion
运行：python3 tr_measurement.py | 输出 tr_report.json
"""
import json,math,re,sys,time
from typing import List,Set,Dict,Tuple
_STOP=frozenset("的了是在我有和就不人都一上也很到说要去你会着没有看好自己"
  "这他她们那被从把让用还可以但而与对为 the a an is are was be have "
  "has do will would could should of in to for with on at by from as it "
  "this that we they he she and or but not no so if than too very just "
  "about up out".split())
def tok(t:str)->Set[str]:
    en=set(re.findall(r'[a-zA-Z_]{2,}',t.lower()))
    ch=[c for c in t if '\u4e00'<=c<='\u9fff']
    return(en|{ch[i]+ch[i+1] for i in range(len(ch)-1)})-_STOP
def sim(a:Set[str],b:Set[str])->float:
    return len(a&b)/len(a|b) if a and b else 0.0
class Store:
    def __init__(self,budget=400,top_k=6):
        self.db:List[dict]=[];self.budget=budget;self.top_k=top_k
    def store(self,c:str,pid:int):
        self.db.append({"id":len(self.db),"c":c,"t":tok(c),"pid":pid})
    def query(self,q:str)->List[dict]:
        qt=tok(q);s=[(sim(qt,m["t"])*.8+.16,m) for m in self.db]
        s.sort(key=lambda x:-x[0]);sel,used=[],0
        for sc,m in s:
            if len(sel)>=self.top_k or sc<.05:break
            if used+len(m["c"])//2>self.budget:continue
            sel.append(m);used+=len(m["c"])//2
        return sel

# ═══ 30组Session Pair（topic|stmts|queries，|分隔紧凑格式）═══
_RAW=[
"REST API版本控制|URL路径版本控制/api/v1/users|语义化版本major.minor.patch|破坏性变更升major|向后兼容升minor|废弃接口保留两个大版本|OpenAPI 3.1规范||接口怎么管理不同版本|破坏性改动怎么处理",
"PostgreSQL vs MySQL|选PostgreSQL因JSONB支持|全文搜索比MySQL快10倍|团队已有PG运维经验|需要JSON存储半结构化数据|MVCC并发更稳定|PG写多读少更合适||关系数据库为什么最终定下来了|并发写入表现怎么样",
"微服务拆分策略|按业务域拆5个微服务|gRPC内部REST外部|每服务独立数据库|Consul服务发现Envoy网关|Saga模式保证一致性|Prometheus+Grafana监控|ELK日志||服务之间怎么互相调用|数据一致性怎么保障|用了什么观测平台",
"Redis缓存策略|Redis Cluster 3主3从|布隆过滤器防穿透|随机过期防雪崩|热key用MEMORY USAGE检测|Cache Aside一致性|删缓存比更新更安全||缓存被击穿了怎么办|数据和缓存不同步怎么处理",
"CI/CD流水线设计|GitHub Actions平台|PR触发测试main触发部署|三层测试|蓝绿部署零停机|保留5个版本镜像|覆盖率>=80%|安全扫描集成CI||自动化发布走的什么流程|上线怎么做到不停服|代码质量有什么卡点",
"WebSocket实时通信|Socket.IO框架|心跳30秒超时60秒断开|RabbitMQ广播|指数退避重连最大30秒|Protobuf序列化|时间戳+序列号去重||客户端掉线了怎么恢复|消息格式用的什么",
"日志系统设计|结构化JSON日志|五级日志级别|生产默认INFO|trace_id链路追踪|热7天温30天冷1年|敏感信息脱敏||日志存多久|密码和token这种怎么处理",
"容器化部署|Alpine基础镜像|多阶段构建|非特权用户|CPU 2核内存2GB|health健康检查|滚动更新maxSurge=1||容器安全怎么保证|资源上限是多少",
"GraphQL vs REST|数据聚合用GraphQL|简单CRUD用REST|N+1用DataLoader|GraphQL需复杂度分析|REST缓存更成熟|内部GraphQL外部REST||对外和对内分别用什么接口风格|查询性能有什么坑",
"分布式锁设计|Redis Redlock|过期=2倍最大执行时间|看门狗10秒续期|公平队列防饥饿|读写锁RWLock|按资源ID加锁||多节点竞争同一资源怎么协调|锁被人抢走了怎么处理",
"Python编码风格|4空格缩进|函数类间空两行|行宽120|优先f-string|类型注解+mypy|isort导入排序|snake_case命名||代码格式有什么统一要求|变量起名有什么规矩",
"Git工作流|Git Flow分支|feature从develop切|type(scope): desc格式|type只用feat/fix等|PR需approve|squash merge||分支怎么管理|提交信息有什么格式规范",
"IDE选择|VS Code|必装Python Pylance GitLens|One Dark Pro主题|WSL2终端|默认快捷键|Settings Sync同步||平时写代码用什么工具|有哪些必装的东西",
"代码审查标准|PR<=400行|审查重点逻辑>性能>风格|风格交给linter|bug引用issue|must-fix和suggestion两类|48小时完成||review有什么标准|一次提太多代码行会怎样",
"测试编写规范|test_xxx.py同目录|一测一行为|test_功能_场景_期望|Mock外部不Mock内部|Docker Compose集成测试|工厂模式生成测试数据||测试代码怎么组织|Mock有什么原则",
"文档编写习惯|README含介绍快速开始API|注释解释为什么|Mermaid架构图|Keep a Changelog|Markdown为主|中文为主API双语||文档有什么要求|注释写不写",
"任务管理工具|Linear管理|5个状态流转|每周一sprint规划|紧急bug直接hotfix|5种标签|<=15 story point/sprint||任务跟踪用什么|一个迭代做多少东西",
"代码重构原则|重构前必须有测试|每次改一个方面|不改变外部行为|先写测试再重构|小步提交可回滚|大重构需设计文档||改老代码有什么规矩|大改动前要做什么准备",
"错误处理风格|自定义异常继承BaseException|异常不吞掉|业务用自定义系统用内置|日志带traceback|用户看到友好错误|重试最多3次指数退避||出错了怎么处理|用户看到的报错长什么样",
"性能优化优先级|先profiling再优化|数据库优化优先|缓存命中率>=80%|P99<200ms|内存<=80%限额|大文件流式处理||性能调优先做什么|响应时间有什么要求",
"选择TypeScript|TS类型安全减40%运行时错误|IDE补全重构好|团队有JS动态类型痛点|React+TS主流|渐进式迁移可接受|维护成本低30%||前端语言最终定的什么|对后续维护有什么影响",
"选择Docker|容器启动比VM快100倍|镜像体积VM的1/10|K8s生态成熟|开发Compose生产K8s|需完全隔离才用VM|团队已有Docker经验||为什么不用虚拟机|开发环境和线上环境怎么区分",
"选择ElasticSearch|倒排索引全文搜索快|ik分词器支持中文|支持模糊搜索评分|PG全文搜索只适合简单|ES可水平扩展PB级|ES延迟10ms PG 200ms||搜索功能为什么单独搞了一套|和直接查数据库比差多少",
"选择Vue|团队Vue经验丰富|单文件组件效率高|响应式比hooks直观|Composition API解决复用|中后台组件库合适|React大SPA移动端更强||前端框架选的哪家|跟另一个主流框架比怎么选",
"选择gRPC|HTTP/2多路复用性能好|Protobuf比JSON小3-5倍|proto强类型接口|自动生成客户端代码|延迟敏感gRPC更低|对外仍用REST||内部通信为什么不用JSON|接口定义怎么管",
"选择Kafka|Kafka吞吐10万+/秒vs RabbitMQ 1万|消息持久化可回放|事件溯源需要log特性|消费者组水平扩展|运维有专人|简单队列仍用RabbitMQ||消息中间件为什么选了那个|不同场景怎么分配",
"选择Tailwind|原子化CSS更灵活|无样式冲突|tree-shaking小体积|团队设计能力强需定制|Bootstrap样式难覆盖|配合PurgeCSS<10KB||样式方案选的什么|跟传统UI库比有什么好处",
"选择Kubernetes|K8s社区大|HPA自动扩缩容成熟|Istio服务网格丰富|云厂商原生支持K8s|Swarm功能不够|K8s是事实标准||容器编排为什么选这个|自动伸缩怎么做",
"选择Monorepo|共享代码不需发npm|跨项目原子重构|CI只构建受影响部分|一个PR看全部变更|版本依赖管理简单|Nx管理构建||代码放一个仓库还是多个|构建怎么处理",
"选择SQLite|零配置嵌入式|FTS5全文搜索内置|WAL模式读写并发够|<10GB性能优于PG|备份就是一个文件|可平滑迁移到PG||本地数据怎么存的|以后数据多了怎么办",
]

def _parse_pairs(raw: List[str]) -> List[Tuple[str, List[str], List[str]]]:
    """解析|分隔的紧凑格式为(topic, stmts, queries)三元组"""
    pairs = []
    for line in raw:
        parts = line.split("||")  # double pipe分隔statements和queries
        topic_stmts = parts[0].split("|", 1)
        topic = topic_stmts[0]
        stmts = topic_stmts[1].split("|") if len(topic_stmts) > 1 else []
        queries = parts[1].split("|") if len(parts) > 1 else []
        pairs.append((topic, stmts, queries))
    return pairs

PAIRS = _parse_pairs(_RAW)

def classify(stoks, ret, budget, top_k, n):
    if ret and sum(1 for m in ret if sim(stoks,m["t"])<.03)>=len(ret)*.5:
        return "retrieval_noise"
    if n>top_k*3: return "context_pressure"
    return "injection_skip"

def run():
    store=Store(budget=400,top_k=6)
    results,ts,tr=[],0,0
    fd:Dict[str,int]={"retrieval_noise":0,"injection_skip":0,
                       "context_pressure":0,"model_confusion":0}
    print("="*60);print("Tr 跨会话记忆迁移率测量实验");print("="*60)
    for i,(topic,stmts,queries) in enumerate(PAIRS):
        pid=i+1
        for s in stmts:store.store(s,pid)
        ts+=len(stmts)
        ret_ids=set()
        for q in queries:
            for m in store.query(q):ret_ids.add(m["id"])
        ret=[m for m in store.db if m["id"] in ret_ids]
        ok,fails=0,[]
        for s in stmts:
            st=tok(s)
            if any(sim(st,m["t"])>.2 for m in ret):ok+=1
            else:
                f=classify(st,ret,store.budget,store.top_k,len(store.db))
                fails.append(f);fd[f]+=1
        tv=ok/len(stmts) if stmts else 0;tr+=ok
        icon="✅" if tv>=.6 else "⚠️" if tv>=.3 else "❌"
        results.append({"pair_id":pid,"topic":topic,"stored":len(stmts),
            "retrieved":ok,"tr":round(tv,3),"failures":fails})
        print(f"  {icon} Pair {pid:2d} {topic:24s} Tr={tv:.0%} ({ok}/{len(stmts)})"
              +(f"  ×{','.join(set(fails))}" if fails else ""))
    overall=tr/ts if ts else 0;passes=overall>=.23
    pri=max(fd,key=lambda k:fd[k]) if any(fd.values()) else "N/A"
    root=f"主要失败模式: {pri} ({fd.get(pri,0)}次)" if pri!="N/A" else "N/A"
    conc="EFFECTIVE" if passes else "INEFFECTIVE"
    report={"total_session_pairs":30,"total_statements":ts,"total_retrieved":tr,
        "tr_rate":round(overall,4),"tr_threshold_23pct":"PASS" if passes else "FAIL",
        "failure_distribution":fd,"per_session":results,
        "conclusion":f"Cross-session transfer {conc}","root_cause":root}
    with open("tr_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"\n{'='*60}")
    print(f"Tr 总迁移率: {overall:.1%}  (阈值23% → {'PASS ✅' if passes else 'FAIL ❌'})")
    print(f"总语句: {ts}, 成功迁移: {tr}")
    print(f"失败分布: {json.dumps(fd,ensure_ascii=False)}")
    print(f"结论: {report['conclusion']}");print(f"根因: {root}")
    print(f"报告: tr_report.json");print("="*60)
    return report

if __name__=="__main__":
    r=run();sys.exit(0 if r["tr_threshold_23pct"]=="PASS" else 1)
