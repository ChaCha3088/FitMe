import csv
from pandas import DataFrame
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def print_hi():
    # 행은 15300개 열은 4219개의 df 생성
    data = DataFrame(0, index=range(15301), columns=range(4220))

    # csv 파일 읽어오기
    with open('1.csv', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split(',')

            # product_id
            product_id = int(line[1])

            # tag_id
            tag_id = int(line[2])

            # data에 값 넣기
            data[tag_id][product_id] = 1

    # 저장
    data.to_pickle('data.pkl')

    # 저장된 pkl 파일 불러오기
    with open('drive/MyDrive/data.pkl', "rb") as file:
        loaded_data = pickle.load(file)

        # 51행을 출력
        print(loaded_data[51])

        # 코사인 유사도 계산
        cos_sim = cosine_similarity(loaded_data, loaded_data)

        # 저장
        cos_sim.to_pickle('cos_sim.pkl')

    # 저장된 pkl 파일 불러오기
    result = []

    with open("drive/MyDrive/Colab Notebooks/cos_sim.pkl", "rb") as file:
        loaded_data = pickle.load(file)

        # 15301행, 15301열의 np.array에서
        # 1부터 15300까지 행을 순회하면서
        for row in range(1, 15301):
            # 1부터 15300까지 열을 대상으로
            indices_and_values = [(i, value) for i, value in enumerate(loaded_data[row])]

            isNoData = True

            # 자기 자신을 제외한 1부터 15300까지
            for i in range(1, 15301):
                # 자기 자신은 제외
                if i == row:
                    continue

                # 자기 자신을 제외한 1부터 15300까지의 value가 0 초과면
                if indices_and_values[i][1] > 0:
                    isNoData = False
                    break

            if isNoData:
                continue

            # 행 번호와 같은 자기 자신은 제외
            indices_and_values = [item for item in indices_and_values if item[0] != row]

            # 맨 앞에 0번째 열은 제외
            indices_and_values = indices_and_values[1:]

            # 유사도가 높은 순서대로 정렬
            indices_and_values.sort(key=lambda x: x[1], reverse=True)

            # 10개까지 리스트에 추가
            result.append([row] + [index for index, value in indices_and_values[:10]])

        # csv 파일로 저장
        with open('drive/MyDrive/Colab Notebooks/result.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
