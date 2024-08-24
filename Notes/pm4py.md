# Getting Started with PM4PY


Process mining is a powerful data-driven approach used to analyze and optimize business processes. By leveraging event logs generated during process execution, this methodology offers a suite of techniques to enhance operational efficiency and effectiveness.

Process mining utilises event logs which are detailed records of process activities timestamp and associated data which helps us analyse and have a better understanding of our business processes

It has some core techniques - 
 - **Process Discovery** - Automatically constructs process models from event logs and helps provide a visual representation of the actual process flows 
 - **Conformance checking** - Compares the discovered process models with predefined or expected models to identify deviations and efficiency 
 - **Process Enhancement** - Extends and improves existing models using the information from event log and real time data.


While most of the event logs are stored in the ERP software such as SAP , Oracle , We can still perform a lot of process mining practices using a python library called PM4PY. 

PM4PY helps us manipulate the event log by creation and modification of petri nets and lot of other process mining algorithms.

Lets take a look at them - 

### Visualisation 

1. Creating a Petri net from an XES file:

```python
import pm4py

# Load the XES file
log = pm4py.read_xes("path/to/your/file.xes")

# Discover Petri net using the inductive miner
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)

# Visualize the Petri net
pm4py.view_petri_net(net, initial_marking, final_marking)
```

This code snippet loads your XES file, discovers a Petri net using the inductive miner algorithm, and visualises the result.

We can also create BPMN that is just another way to represent process flows like petri nets here by just using `pm4py.discover_bpmn_inductive` 

### Performance Analysis

2. We can find the details of the case duration of our event log using 

```python 
case_duration = pm4py.get_all_case_durations(df)
print(case_duration)
```

We get something like - 
```python 
[Timedelta('0 days 00:00:00'), Timedelta('0 days 00:00:08'), Timedelta('0 days 00:00:09'), Timedelta('0 days 00:00:09'), Timedelta(....
```

3. The way we got case duration for all cases we can also find out the start time of activities using : `start_activities = pm4py.get_start_activities(log)` Get end activities: `end_activities = pm4py.get_end_activities(log)`. Giving Results like - 

```python 
{'1': 3644, '3': 108, '8': 48, '6': 2, '2': 1, '9': 1}
{'6': 3804}
```
Respectively 

### Filtering 

4. We can perform both performance based and time based filtering of the event logs - 
```python 
# for filtering based on time
filtered_log = pm4py.filter_time_range(df, "2023-01-01 00:00:00", "2023-12-31 23:59:59")

# for filtering  based on performance filters
filtered_log = pm4py.filter_case_performance(df, min_performance=100, max_performance=1000)
```

5. `min_performance=100` sets the minimum performance threshold to 100 (units depend on your data, typically seconds) `max_performance=1000` sets the maximum performance threshold to 1000 . 
6. `Performance` here in this context typically refers to the case duration which is the time difference between the first and the last event of a case.


### Charts 

7. We can also create dotted charts to view the performance of the event log over time by using `pm4py.view_dotted_chart(xesfile)` where the y axis denotes case index and the x axis denotes time.
```python 
pm4py.view_dotted_chart(xesfile)
```
8. Similarly we can also create a process map by utilising the `pm4py.view_process_map(df)` function.

### Statistics 

9. We can perform statistics such as average arrival time of case using functions such as - 
```python 
case_arrival = pm4py.get_case_arrival_average(df)

# output - 0 days 06:14:34
```
10. We can also calculate events per case using : `events_per_case = pm4py.get_events_per_case(log)` 
```python 

```
11. Finally we can convert our XES file to CSV or vice versa and save the changes using 
```python 
pm4py.write_xes(log, "output.csv")
```


## Conclusion 

Hence we have practised the basics of playing around with PM4PY , while there is another big area yet to be explored such as Object Centric Process Mining , This acts as a starting point of your Process Mining journey. 

I will be posting more learnings around process mining .

You can follow me on Twitter , LinkdIn

Thanks For Reading !

