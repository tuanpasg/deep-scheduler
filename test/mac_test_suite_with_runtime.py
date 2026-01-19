import time
import random
import statistics

class Scheduler:
    def schedule(self, ue_list, tti):
        # Dummy scheduling logic
        return random.choice(ue_list) if ue_list else None

class RoundRobinScheduler(Scheduler):
    def __init__(self):
        self.index = 0

    def schedule(self, ue_list, tti):
        if not ue_list:
            return None
        ue = ue_list[self.index % len(ue_list)]
        self.index += 1
        return ue

class ProportionalFairScheduler(Scheduler):
    def schedule(self, ue_list, tti):
        if not ue_list:
            return None
        # PF metric approximated with random selection weighted by load
        return max(ue_list, key=lambda u: random.random() * u["load_bytes"])

class SchedulerTestSuite:
    def __init__(self, schedulers, num_ttis=1000):
        self.schedulers = schedulers
        self.num_ttis = num_ttis
        self.metrics = {name: {"throughputs": [], "fairness": [], "runtimes": []}
                        for name in schedulers.keys()}

    def run(self, ue_generator):
        for tti in range(self.num_ttis):
            ue_list = ue_generator()
            for name, scheduler in self.schedulers.items():
                start_time = time.perf_counter()
                selected = scheduler.schedule(ue_list, tti)
                elapsed = time.perf_counter() - start_time

                self.metrics[name]["runtimes"].append(elapsed)

                if selected:
                    throughput = selected["mcs_idx"] * random.randint(1, 10)
                    self.metrics[name]["throughputs"].append(throughput)
                    fairness_val = throughput / (selected["load_bytes"] + 1)
                    self.metrics[name]["fairness"].append(fairness_val)

    def report(self):
        report = {}
        for name, metric in self.metrics.items():
            avg_tp = statistics.mean(metric["throughputs"]) if metric["throughputs"] else 0
            avg_fairness = statistics.mean(metric["fairness"]) if metric["fairness"] else 0
            avg_runtime = statistics.mean(metric["runtimes"]) * 1e6 if metric["runtimes"] else 0  # us
            report[name] = {
                "avg_throughput": avg_tp,
                "avg_fairness": avg_fairness,
                "avg_runtime_us": avg_runtime
            }
        return report

def ue_generator():
    num_ues = random.randint(1, 10)
    return [{"id": i, "load_bytes": random.randint(100, 1000), "mcs_idx": random.randint(1, 28)}
            for i in range(num_ues)]

if __name__ == "__main__":
    schedulers = {
        "RR": RoundRobinScheduler(),
        "PF": ProportionalFairScheduler()
    }
    suite = SchedulerTestSuite(schedulers, num_ttis=500)
    suite.run(ue_generator)
    results = suite.report()
    for name, stats in results.items():
        print(f"Scheduler: {name}")
        print(f"  Avg Throughput: {stats['avg_throughput']:.2f}")
        print(f"  Avg Fairness: {stats['avg_fairness']:.4f}")
        print(f"  Avg Runtime: {stats['avg_runtime_us']:.2f} µs")
