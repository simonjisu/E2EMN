import torch
from memn2n.utils import Config
from memn2n.train import import_data, build_model, validation

def main(train_type, emd=20):
    weight_styles = ['adjacent', 'rnnlike']
    other_methods = ['bow', 'pe_te']
    config = Config(emd=emd)
    device = None
    map_location = {'cuda:0':'cpu'} if device is None else None
    write_file = open('test_result', 'w', encoding='utf-8')
    print('| weight style | method | error rate |', file=write_file)
    print('|--|--|--|', file=write_file)
    for w_style in weight_styles:
        for method in other_methods:
            for i in range(1, 21):
                load_path = config.build(task=i, weight_style=w_style, other_method=method)
                test, test_loader = import_data(config, device, is_test=True)
                memn2n, loss_f, *_ = build_model(config, test.vocab, test.maxlen_story, device)
                memn2n.load_state_dict(torch.load(load_path,  map_location=map_location))
                test_loss, error_rate = validation(test.vocab, test_loader, memn2n, loss_f, 
                                                   is_test=True)
                
                print('| {} | {} | {:.4f} |'.format(w_style, method, error_rate), file=write_file)
    write_file.close()

if sys.argv[1] == '-h':
    print('Set test type by insert argument "-type [some type]"')
    print('> For learned all tasks jointly, type "jnt".[not ready: fixing vocab issue]')
    print('> For learned all tasks independently, type "ind".')
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