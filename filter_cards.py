import json
import sys
import ijson
import pandas

def main():
    if len(sys.argv) < 4:
        print("Usage: python filter_cards.py usage input.json output.json")
        sys.exit(2)

    usage = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    if usage == "commander_legal":
        commander_legal(input_file, output_file)
    elif usage == "filter_attributes":
        filter_attributes(input_file, output_file)
    elif usage == "convert_to_csv":
        convert_to_csv(input_file, output_file)
    elif usage == "guilds_of_ravnica":
        guilds_of_ravnica(input_file, output_file)
    elif usage == "change_colors":
        print("function called")
        change_colors(input_file, output_file)
        print("done")
    else:
        sys.exit("didn't work")

def guilds_of_ravnica(input_file, output_file):
    data = pandas.read_csv(input_file)
    data = data[data['set_name'] == 'Guilds of Ravnica']
    data.to_csv(output_file, index=False)

def change_colors(input_file, output_file):
    df = pandas.read_csv(input_file)
    df.loc[df['colors'].astype(str) == "nan", 'colors'] = 'C'
    df.loc[df['colors'].str.len() > 1, 'colors'] = 'M'
    df.to_csv(output_file)

def convert_to_csv(input_file, output_file):
    first = True
    with open(input_file, 'r', encoding='utf-8') as f, \
        open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('name,art_crop,cmc,type_line,colors,set_name,rarity\n')
        for item in ijson.items(f, 'item'):
            if not first:
                out_f.write('\n')
            first = False

            if isinstance(item, dict):
                name = item.get('name', '').replace('"', '""')
                art_crop = item.get('art_crop', '').replace('"', '""')
                cmc = item.get('cmc', '').replace('"', '""')
                type_line = item.get('type_line', '').replace('"', '""')
                colors = ','.join(item.get('colors', []))
                set_name = item.get('set_name', '').replace('"', '""')
                rarity = item.get('rarity', '').replace('"', '""')

                out_f.write(f'"{name}","{art_crop}","{cmc}","{type_line}","{colors}","{set_name}","{rarity}"')

def filter_attributes(input_file, output_file):
    desired_attributes = {
        'name',
        'type_line',
        'colors',
        'set_name',
        'rarity',
        'cmc',
        'image_uris', 
    }

    first = True
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('[\n')
        for item in ijson.items(f, 'item'):
            if isinstance(item, dict):
                filtered_item = {}
                for k, v in item.items():
                    if k in desired_attributes:
                        if k=='image_uris':
                            v = v['art_crop']
                            k = 'art_crop'
                        filtered_item[k] = v

            else:
                filtered_item = item

            if not first:
                out_f.write(',\n')

            json.dump(filtered_item, out_f, ensure_ascii=False, default=str)

            first = False

        out_f.write('\n]\n')

def commander_legal(input_file, output_file):
    first = True
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('[\n')
        for item in ijson.items(f, 'item'):

            commander_status = None
            if isinstance(item, dict):
                legalities = item.get('legalities')
                if isinstance(legalities, dict):
                    commander_status = legalities.get('commander')

            if commander_status == 'not_legal':
                continue

            if not first:
                out_f.write(',\n')
            
            json.dump(item, out_f, ensure_ascii=False, default=str)

            first = False

        out_f.write('\n]\n')

if __name__ == "__main__":
    main()
