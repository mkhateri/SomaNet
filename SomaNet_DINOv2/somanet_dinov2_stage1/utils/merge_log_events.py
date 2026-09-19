## merge training and resuming logged events
import os
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

def combine_event_files(log_dir, output_file):
    """Combine all event files in a directory into a single event file."""
    with tf.summary.create_file_writer(output_file).as_default() as writer:
        for root, _, files in os.walk(log_dir):
            for filename in sorted(files):
                if "events.out.tfevents" in filename:
                    event_file = os.path.join(root, filename)
                    for event in summary_iterator(event_file):
                        for value in event.summary.value:
                            if value.HasField('simple_value'):
                                tf.summary.scalar(value.tag, value.simple_value, step=event.step)
                            elif value.HasField('histo'):
                                tf.summary.histogram(value.tag, value.histo, step=event.step)
                            # Add additional checks for other types of summaries if needed
        writer.flush()

if __name__ == "__main__":

    log_dir = './20240721-045246'  # Directory containing the event files
    output_file = './20240721-045246/combined_events'  # Path to the combined event file

    combine_event_files(log_dir, output_file)