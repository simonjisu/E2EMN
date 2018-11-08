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

def get_command_string(task, emd, wstyle, encmth, temporal, save_path, log_path):
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
    -svp "{}" > {}'''.format(task, emd, wstyle, encmth, temporal, save_path, log_path)
    command_str = [x.strip() for x in command_str.split('\n')]
    return ' '.join(command_str)

def main(emd):
    weight_styles = ['adjacent', 'rnnlike']
    other_methods = ['bow', 'pe_te']
    for path in ['./trainlog', './saved_models']:
        check_build_dir(path, weight_styles, other_methods)
    if emd == 20:
        for wstyle in weight_styles:
            for enc_method in enc_methods:
                for task in range(1, 21):
                    encmth = 'pe' if len(enc_method.split('_')) > 1 else 'bow'
                    temporal = "-temporal" if encmth == 'pe' else ""
                    save_path = os.path.join('./saved_models/', wstyle, enc_method, 'task{}.model'.format(task))
                    log_path = os.path.join('./trainlog/', wstyle, enc_method, 'task{}.log'.format(task))
                    command_str = get_command_string(task, emd, wstyle, encmth, temporal, save_path, log_path)
                    print('--'*5, 'task: {}'.format(task), wstyle, encmth, temporal.strip('-')[:2], '--'*5)
                    print(command_str)
                    os.system(command_str)
                    print('done!')
    elif emd == 50:
        assert True, '[not ready: fixing vocab issue]
    else:
        print('error')
        

if sys.argv[1] == '-h':
    print('Set embedding demension by insert argument "-emd [some number]"')
    print('> For learning all tasks jointly use 50.[not ready: fixing vocab issue]')
    print('> For learning all tasks independently use 20.')
elif sys.argv[1] == '-emd':
    emd = int(sys.argv[2])
    main(emd)
else:
    print('Argument Error: type "-h" for help')