import pandas as pd
from pathlib import Path


def process_files(input_dir, output_dir, record_name):

    img_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    record_path = output_dir / record_name
    class_path = output_dir / 'classes.names'

    img_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    copy_images(input_dir, img_dir)

    copy_labels(input_dir, labels_dir, record_path, class_path)


def copy_images(input_dir, img_dir):

    for input_img_path in input_dir.glob('*png'):
        img_path = img_dir / input_img_path.name
        print('Writing', img_path)
        img_path.write_bytes(input_img_path.read_bytes())


def copy_labels(input_dir, labels_dir, record_path, class_path):

    input_labels_path = input_dir / 'labels.csv'

    df = pd.read_csv(input_labels_path)

    class_names = df['class'].unique()

    class_ids = dict(zip(class_names, range(len(class_names))))

    df['class_id'] = df['class'].map(class_ids)

    # write class ids to file
    with open(class_path, 'w') as class_file:
        for class_name in class_ids.keys():
            class_file.write(f'{class_name}\n')


    # write box coordinates to files
    with open(record_path, 'w') as record_file:

        print('Writing', record_path)

        for input_filename, dfg in df.groupby('filename'):

            labels_path = labels_dir / Path(input_filename).with_suffix('.txt')

            # write all boxes to a single file 
            with open(labels_path, 'w') as labels_file:

                print('Writing', labels_path)

                for _, row in dfg.iterrows():
                    labels_file.write(convert_boxes(row))

            # add image filename to record
            record_file.write(f'data/images/{input_filename}\n')

    


def convert_boxes(row):
    ''' Extract box coordinates from dataframe row '''
    class_id = row['class_id']
    x_center = (row['xmax'] + row['xmin']) * 0.5 / row['width']
    y_center = (row['ymax'] + row['ymin']) * 0.5 / row['height']
    width = (row['xmax'] - row['xmin']) / row['width']
    height = (row['ymax'] - row['ymin']) / row['height']

    return f'{class_id} {x_center} {y_center} {width} {height}\n'


if __name__ == '__main__':


    process_files(
        input_dir=Path('tensorflow/data/train'),
        output_dir=Path('pytorch/data'),
        record_name='train.txt'
    )

    process_files(
        input_dir=Path('tensorflow/data/test'),
        output_dir=Path('pytorch/data'),
        record_name='test.txt'
    )
