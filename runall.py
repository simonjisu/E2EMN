# run all files
import os
import sys

# build trainlog
def check_build_dir(path, weight_styles, other_methods):
    if os.path.isdir(path):
        for x in weight_styles:
            weight_style_path = os.path.join(path, x)
            if not os.path.isdir(weight_style_path):
                os.mkdir(weight_style_path)
            for y in other_methods:
                methods_path = os.path.join(path, x, y)
                if not os.path.isdir(methods_path):
                    os.mkdir(methods_path)
                    print(methods_path, 'done!')
                else:
                    print(methods_path, 'exists')
    else:
        os.mkdir(path)
        check_build_dir(path, weight_styles, other_methods)

def get_command_string(task, emd, wstyle, mth, temporal, save_path, log_path):
    command_str = \
    '''python3 -u ./memn2n/main.py
    -root "./data/QA_bAbI_tasks/en-valid-10k/"
    -task {}
    -bs 32
    -cuda
    -emd {}
    -wstyle "{}"
    -encmth "{}"
    {}
    -thres 5
    -lr 0.01
    -stp 100
    -anl 0.5
    -save
    -savebest
    -svp "{}" > {}'''.format(task, emd, wstyle, mth, temporal, save_path, log_path)
    command_str = [x.strip() for x in command_str.split('\n')]
    return ' '.join(command_str)

def main(train_type, emd=100):
    weight_styles = ['adjacent', 'rnnlike']
    other_methods = ['bow', 'pe_te']
    for path in ['./trainlog', './saved_models']:
        check_build_dir(path, weight_styles, other_methods)
    if train_type == 'ind':
        for wstyle in weight_styles:
            for method in other_methods:
                for task in range(1, 21):
                    mth = 'pe' if 'pe' in method.split('_') else 'bow'
                    temporal = "-temporal" if 'te' in method.split('_') else ""
                    save_path = os.path.join('./saved_models/', wstyle, method, 'task{}.model'.format(task))
                    log_path = os.path.join('./trainlog/', wstyle, method, 'task{}.log'.format(task))
                    command_str = get_command_string(task, emd, wstyle, mth, temporal, save_path, log_path)
                    print('--'*5, 'task: {}'.format(task), wstyle, method, '--'*5)
                    print(command_str)
                    os.system(command_str)
                    print('done!')
    elif train_type == 'jnt':
        assert True, '[not ready: fixing vocab issue]'
    else:
        print('error')
        

if sys.argv[1] == '-h':
    print('Set train type by insert argument "-type [some type]"')
    print('> For learning all tasks jointly, type "jnt".[not ready: fixing vocab issue]')
    print('> For learning all tasks independently, type "ind".')
    print('Set embedding demension by insert argument after "-type", "-emd [number]", else will be 100')
elif sys.argv[1] == '-type':
    train_type = sys.argv[2]
    if sys.argv[3] == '-emd':
        emd = int(sys.argv[4])
    else:
        emd = 100
    main(train_type, emd=emd)
else:
    print('Argument Error: type "-h" for help')