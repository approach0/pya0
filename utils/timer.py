import json
import time
import statistics

timer_records = []
start = None

def timer_begin():
    global start
    start = time.time()

def timer_end():
    global start
    end = time.time()
    delta = end - start
    print('delta time:', delta)
    timer_records.append(delta)

def timer_report(report_filename='timer_report.json'):
    global timer_records
    output = {}
    runtimes = timer_records
    output['avg'] = statistics.mean(runtimes)
    output['med'] = statistics.median(runtimes)
    output['max'] = max(runtimes)
    output['min'] = min(runtimes)
    output['len'] = len(runtimes)
    if len(runtimes) >= 2:
        output['std'] = statistics.stdev(runtimes)
    for key in output:
        output[key] = round(output[key], 3) # to milli-seconds
    print('timer_report:', output)
    output['_runtimes'] = runtimes
    output_json = json.dumps(output, sort_keys=True, indent=4)
    with open(report_filename, 'w') as fh:
        fh.write(output_json)
        fh.write('\n')
