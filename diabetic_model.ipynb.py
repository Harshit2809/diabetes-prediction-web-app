{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "diabetic model.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "J7mIXBcARatF"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn import svm\n",
        "from sklearn.metrics import accuracy_score\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HB06kKeUS0Mq"
      },
      "source": [
        "Data collection and analysis\n",
        "\n",
        "Pima diabetes dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZRT6YMpTS5Iz"
      },
      "source": [
        "#loading the data set to a pandas dataframe\n",
        "diabetes_dataset = pd.read_csv('/content/diabetes.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gDQMGwbZVJuO"
      },
      "source": [
        "printing first five rows of the dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "g36obMLaVRSW",
        "outputId": "799c269f-8b1c-4f60-b1d2-77830d15990f"
      },
      "source": [
        "diabetes_dataset.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "      <th>Outcome</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>6</td>\n",
              "      <td>148</td>\n",
              "      <td>72</td>\n",
              "      <td>35</td>\n",
              "      <td>0</td>\n",
              "      <td>33.6</td>\n",
              "      <td>0.627</td>\n",
              "      <td>50</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>85</td>\n",
              "      <td>66</td>\n",
              "      <td>29</td>\n",
              "      <td>0</td>\n",
              "      <td>26.6</td>\n",
              "      <td>0.351</td>\n",
              "      <td>31</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>8</td>\n",
              "      <td>183</td>\n",
              "      <td>64</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>23.3</td>\n",
              "      <td>0.672</td>\n",
              "      <td>32</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>89</td>\n",
              "      <td>66</td>\n",
              "      <td>23</td>\n",
              "      <td>94</td>\n",
              "      <td>28.1</td>\n",
              "      <td>0.167</td>\n",
              "      <td>21</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>137</td>\n",
              "      <td>40</td>\n",
              "      <td>35</td>\n",
              "      <td>168</td>\n",
              "      <td>43.1</td>\n",
              "      <td>2.288</td>\n",
              "      <td>33</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome\n",
              "0            6      148             72  ...                     0.627   50        1\n",
              "1            1       85             66  ...                     0.351   31        0\n",
              "2            8      183             64  ...                     0.672   32        1\n",
              "3            1       89             66  ...                     0.167   21        0\n",
              "4            0      137             40  ...                     2.288   33        1\n",
              "\n",
              "[5 rows x 9 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FS2Sw9B3Vl6C",
        "outputId": "c6eeb9a5-180e-49ed-d121-57a62cd74e9c"
      },
      "source": [
        "# number of rows and columns in this dataset\n",
        "diabetes_dataset.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(768, 9)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "aXsHD3JlWAoR",
        "outputId": "5223f263-d605-42cc-b624-a3b0eea3db95"
      },
      "source": [
        "#getting the statistical measures of the data\n",
        "diabetes_dataset.describe()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "      <th>Outcome</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "      <td>768.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>3.845052</td>\n",
              "      <td>120.894531</td>\n",
              "      <td>69.105469</td>\n",
              "      <td>20.536458</td>\n",
              "      <td>79.799479</td>\n",
              "      <td>31.992578</td>\n",
              "      <td>0.471876</td>\n",
              "      <td>33.240885</td>\n",
              "      <td>0.348958</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>3.369578</td>\n",
              "      <td>31.972618</td>\n",
              "      <td>19.355807</td>\n",
              "      <td>15.952218</td>\n",
              "      <td>115.244002</td>\n",
              "      <td>7.884160</td>\n",
              "      <td>0.331329</td>\n",
              "      <td>11.760232</td>\n",
              "      <td>0.476951</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.078000</td>\n",
              "      <td>21.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>99.000000</td>\n",
              "      <td>62.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>27.300000</td>\n",
              "      <td>0.243750</td>\n",
              "      <td>24.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>3.000000</td>\n",
              "      <td>117.000000</td>\n",
              "      <td>72.000000</td>\n",
              "      <td>23.000000</td>\n",
              "      <td>30.500000</td>\n",
              "      <td>32.000000</td>\n",
              "      <td>0.372500</td>\n",
              "      <td>29.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>6.000000</td>\n",
              "      <td>140.250000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>32.000000</td>\n",
              "      <td>127.250000</td>\n",
              "      <td>36.600000</td>\n",
              "      <td>0.626250</td>\n",
              "      <td>41.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>17.000000</td>\n",
              "      <td>199.000000</td>\n",
              "      <td>122.000000</td>\n",
              "      <td>99.000000</td>\n",
              "      <td>846.000000</td>\n",
              "      <td>67.100000</td>\n",
              "      <td>2.420000</td>\n",
              "      <td>81.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "       Pregnancies     Glucose  ...         Age     Outcome\n",
              "count   768.000000  768.000000  ...  768.000000  768.000000\n",
              "mean      3.845052  120.894531  ...   33.240885    0.348958\n",
              "std       3.369578   31.972618  ...   11.760232    0.476951\n",
              "min       0.000000    0.000000  ...   21.000000    0.000000\n",
              "25%       1.000000   99.000000  ...   24.000000    0.000000\n",
              "50%       3.000000  117.000000  ...   29.000000    0.000000\n",
              "75%       6.000000  140.250000  ...   41.000000    1.000000\n",
              "max      17.000000  199.000000  ...   81.000000    1.000000\n",
              "\n",
              "[8 rows x 9 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pBYUKohoWR3b",
        "outputId": "2699f7ee-c323-4090-e8e4-b88a3b704ccd"
      },
      "source": [
        "diabetes_dataset['Outcome'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    500\n",
              "1    268\n",
              "Name: Outcome, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PlyxRF0MW7Re"
      },
      "source": [
        "0 --> non-diabetic\n",
        "\n",
        "1 --> diabetic"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "id": "9Ey3O4h4WtnQ",
        "outputId": "0c8a1b6c-9ba9-406d-9f6a-51b3482b7824"
      },
      "source": [
        "diabetes_dataset.groupby('Outcome').mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Outcome</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>3.298000</td>\n",
              "      <td>109.980000</td>\n",
              "      <td>68.184000</td>\n",
              "      <td>19.664000</td>\n",
              "      <td>68.792000</td>\n",
              "      <td>30.304200</td>\n",
              "      <td>0.429734</td>\n",
              "      <td>31.190000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4.865672</td>\n",
              "      <td>141.257463</td>\n",
              "      <td>70.824627</td>\n",
              "      <td>22.164179</td>\n",
              "      <td>100.335821</td>\n",
              "      <td>35.142537</td>\n",
              "      <td>0.550500</td>\n",
              "      <td>37.067164</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "         Pregnancies     Glucose  ...  DiabetesPedigreeFunction        Age\n",
              "Outcome                           ...                                     \n",
              "0           3.298000  109.980000  ...                  0.429734  31.190000\n",
              "1           4.865672  141.257463  ...                  0.550500  37.067164\n",
              "\n",
              "[2 rows x 8 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0Gpe3xAoXVHZ"
      },
      "source": [
        "#seperating the data and labels\n",
        "x=diabetes_dataset.drop(columns='Outcome',axis=1)\n",
        "y=diabetes_dataset['Outcome']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RilvBjjrYXAL",
        "outputId": "6392b185-09c0-43b5-813e-ba68a589e96e"
      },
      "source": [
        "print(x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     Pregnancies  Glucose  BloodPressure  ...   BMI  DiabetesPedigreeFunction  Age\n",
            "0              6      148             72  ...  33.6                     0.627   50\n",
            "1              1       85             66  ...  26.6                     0.351   31\n",
            "2              8      183             64  ...  23.3                     0.672   32\n",
            "3              1       89             66  ...  28.1                     0.167   21\n",
            "4              0      137             40  ...  43.1                     2.288   33\n",
            "..           ...      ...            ...  ...   ...                       ...  ...\n",
            "763           10      101             76  ...  32.9                     0.171   63\n",
            "764            2      122             70  ...  36.8                     0.340   27\n",
            "765            5      121             72  ...  26.2                     0.245   30\n",
            "766            1      126             60  ...  30.1                     0.349   47\n",
            "767            1       93             70  ...  30.4                     0.315   23\n",
            "\n",
            "[768 rows x 8 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "M7T-S70yYbTO",
        "outputId": "0565998b-8817-4cc0-a956-0bff29e3aa58"
      },
      "source": [
        "print(y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0      1\n",
            "1      0\n",
            "2      1\n",
            "3      0\n",
            "4      1\n",
            "      ..\n",
            "763    0\n",
            "764    0\n",
            "765    0\n",
            "766    1\n",
            "767    0\n",
            "Name: Outcome, Length: 768, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uwHZbwYvYuGZ"
      },
      "source": [
        "Data standardization"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2k850IKtYm_Y"
      },
      "source": [
        "scaler = StandardScaler()\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5J2xG-vOZt7J",
        "outputId": "7692adc0-1eae-4b56-e164-21965b37d3f9"
      },
      "source": [
        "scaler.fit(x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "StandardScaler()"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HXZJdtjgZxRs"
      },
      "source": [
        "standardizes_data = scaler.transform(x)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VFU6t7VAaFE_",
        "outputId": "40b0eede-7089-4744-fade-5506ec52ac75"
      },
      "source": [
        "print(standardizes_data)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198\n",
            "   1.4259954 ]\n",
            " [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078\n",
            "  -0.19067191]\n",
            " [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732\n",
            "  -0.10558415]\n",
            " ...\n",
            " [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336\n",
            "  -0.27575966]\n",
            " [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101\n",
            "   1.17073215]\n",
            " [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505\n",
            "  -0.87137393]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nMvY3Yd_aI24"
      },
      "source": [
        "x=standardizes_data\n",
        "y=diabetes_dataset['Outcome']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fnmvinDeaqrF",
        "outputId": "7859e743-e2a6-4e54-d353-e0c837f6414d"
      },
      "source": [
        "print(x)\n",
        "print(y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198\n",
            "   1.4259954 ]\n",
            " [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078\n",
            "  -0.19067191]\n",
            " [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732\n",
            "  -0.10558415]\n",
            " ...\n",
            " [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336\n",
            "  -0.27575966]\n",
            " [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101\n",
            "   1.17073215]\n",
            " [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505\n",
            "  -0.87137393]]\n",
            "0      1\n",
            "1      0\n",
            "2      1\n",
            "3      0\n",
            "4      1\n",
            "      ..\n",
            "763    0\n",
            "764    0\n",
            "765    0\n",
            "766    1\n",
            "767    0\n",
            "Name: Outcome, Length: 768, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DLUqcelZbFOA"
      },
      "source": [
        "train test split"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vnUnGjnnbAtP"
      },
      "source": [
        "x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , stratify=y , random_state=2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8vQr9e-qcvT8",
        "outputId": "91291cca-84e0-409e-c62f-795e5099dbe2"
      },
      "source": [
        "print(x.shape,x_train.shape,x_test.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(768, 8) (614, 8) (154, 8)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TDcliLaxc-H6"
      },
      "source": [
        "Training the model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3hxpoZRtc6T_"
      },
      "source": [
        "classifier = svm.SVC(kernel='linear')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ISykOzDBdi28",
        "outputId": "a55803ed-1aba-45ed-d5e5-43f1e0040a5b"
      },
      "source": [
        "#training svm-support vector machine classfier\n",
        "classifier.fit(x_train,y_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC(kernel='linear')"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kaU_nwIJd-I9"
      },
      "source": [
        "model evaluation\n",
        "\n",
        "accuracy score"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7JLHA6HXd4SU"
      },
      "source": [
        "#accuracy score of the training data\n",
        "x_train_prediction = classifier.predict(x_train)\n",
        "training_data_accuracy = accuracy_score(x_train_prediction,y_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "K3f5BHFQe2Ns",
        "outputId": "a461165c-6bcc-4e6f-979a-50cf2a61fd66"
      },
      "source": [
        "print('accuracy score of the training data : ',training_data_accuracy)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "accuracy score of the training data :  0.7866449511400652\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xYP0kQORfCGH"
      },
      "source": [
        "#accuracy score of the testing data\n",
        "x_test_prediction = classifier.predict(x_test)\n",
        "testing_data_accuracy = accuracy_score(x_test_prediction,y_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P__Ix4Ryfyta",
        "outputId": "2241f1de-51f2-4a16-c6fd-3833a1a6f89e"
      },
      "source": [
        "print('accuracy score of the testing data : ',testing_data_accuracy)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "accuracy score of the testing data :  0.7727272727272727\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wxCxeHzmgkrJ"
      },
      "source": [
        "making a predictive system"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bq7Bx_oGgA-5",
        "outputId": "71cc853f-90e0-45e7-99c9-b0dc95041871"
      },
      "source": [
        "input_data = (5,166,72,19,175,25.8,0.587,51)\n",
        "\n",
        "#changing the input data to numpy array\n",
        "input_data_as_numpy_array = np.asarray(input_data)\n",
        "\n",
        "#reshape the array as we are predicting for one instance\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "\n",
        "#standardize the input data\n",
        "std_data = scaler.transform(input_data_reshaped)\n",
        "print(std_data)\n",
        "\n",
        "prediction = classifier.predict(std_data)\n",
        "\n",
        "if prediction == 0 :\n",
        "  print(\"the person is non diabetic\")\n",
        "else:\n",
        "  print(\"the person is diabetic\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 0.3429808   1.41167241  0.14964075 -0.09637905  0.82661621 -0.78595734\n",
            "   0.34768723  1.51108316]]\n",
            "the person is diabetic\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/base.py:446: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
            "  \"X does not have valid feature names, but\"\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ck4QL49tjQUr"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}