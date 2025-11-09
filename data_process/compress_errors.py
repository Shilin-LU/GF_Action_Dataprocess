import os

log_files = [x for x in os.listdir("./") if x.startswith("cuda")]
log_files.sort()
vfr_error_head = "Error in worker thread: The decoder returned a frame that is past the expected"
frame_error_head = 'Assert on "end_frame <= file.frame_count_" failed'
json_error_head = "Failed to dump"
vfr_lines, frame_lines, json_lines, error_set = [], [], [], set()
for log_file in log_files:
    for line in open(log_file).readlines():
        vfr_lines.append(line) if line.startswith(vfr_error_head) else None
        frame_lines.append(line) if line.startswith(frame_error_head) else None
        json_lines.append(line) if line.startswith(json_error_head) else None
with open("2323.log" , "w") as f:
    for line in vfr_lines + frame_lines + json_lines:
        if line not in error_set:
            f.write(line), error_set.add(line)
