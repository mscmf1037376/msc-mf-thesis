import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from wealth_optimizer.simulated_bm import geometric_brownian_motion_generator, generate_simulated_price_paths

if __name__ == '__main__':

    scenarios = [
        {'number': 1,
         'drifts': [0.01, 0.02],
         'vols': [0.2, 0.4],
         'rho': 0.7
         },
        {'number': 2,
         'drifts': [0.01, 0.03],
         'vols': [0.15, 0.20],
         'rho': 0.5
         }
    ]

    # scenarios = [
    #     {   'number': 1,
    #         'drifts': [0.05, 0.1],
    #         'vols': [0.2, 0.4]
    #     },
    #     {'number': 2,
    #      'drifts': [0.0, 0.001],
    #      'vols': [0.2, 0.4]
    #      },
    #     {'number': 3,
    #      'drifts': [0.0, 0.0],
    #      'vols': [0.25, 0.50]
    #      },
    #     {'number': 4,
    #      'drifts': [0.15, 0.2],
    #      'vols': [0.35, 0.55]
    #      },
    #     {'number': 5,
    #      'drifts': [0.0, 0.0],
    #      'vols': [0.2, 0.2]
    #      },
    #     {'number': 6,
    #      'drifts': [0.0, 0.03],
    #      'vols': [0.2, 0.25]
    #      },
    # ]
    sns.set()


    for scenario in scenarios:

        drifts = scenario['drifts']
        print(drifts)

        vols = scenario['vols']
        scenario_number = scenario['number']
        rho = scenario.get('rho', 0.0)
        initial_values = [1.0] * len(drifts)
        maturity = 100
        num_paths = 1
        num_samples = 52 * 100
        num_processes = len(drifts)

        generators = []
        correlation_matrix = np.eye(num_processes)
        correlation_matrix[0][1] = rho
        correlation_matrix[1][0] = rho
        print(correlation_matrix)


        for mu, sigma in list(zip(drifts, vols)):
            generators.append(geometric_brownian_motion_generator(drift=mu, volatility=sigma))

        spots = np.array(initial_values)

        asset_paths = generate_simulated_price_paths(spots, generators, maturity, num_samples, num_paths,
                                                     correlation_matrix)
        # f, subPlots = plt.subplots(2, sharex=True)
        # for i in range(len(uncorrelated_gbms)):
        #     for j in range(num_paths):
        #         subPlots[i].plot(asset_paths[i, j, :])
        # plt.show()

        for j in range(num_paths):
            current_path = asset_paths[:, j, :]
            df = pd.DataFrame(current_path.transpose())
            df.columns = ['asset_' + str(i + 1) for i in range(num_processes)]

            df.plot()
            plt.title('Path j={}'.format(j))
            plt.savefig('two_uncorrelated_gbms_scenario_{}.png'.format(scenario_number))
            returns = (df.diff() / df).dropna(inplace=False)
            returns.index = np.arange(maturity / num_samples, maturity, maturity / num_samples)
            returns.columns = ['asset_' + str(i + 1) for i in range(num_processes)]
            assert isinstance(returns, pd.DataFrame)
            # print(returns)
            #
            returns.to_excel('two_correlated_gbms_scenario_{}.xlsx'.format(scenario_number))
            returns.plot()
            plt.savefig('two_correlated_gbms_scenario_{}_returns.png'.format(scenario_number))


            #
            # np.arange()
            #
            # returns_dataframe = pd.DataFrame(returns, index=np.linspace(maturity/num_steps, maturity, num_steps),
            #                                  columns=['asset_' + str(i + 1) for i in range(num_assets)])
            # print(returns_dataframe)
            # returns_dataframe.plot()
            # returns.hist(bins=np.linspace(-0.2, 0.2, 100))
        plt.show()
