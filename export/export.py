import numpy as np
import shutil

class result_export():
    def __init__(self, data_dict, ratio, velocity, Tin, model, folder_name):

        self.data_dict = data_dict
        self.ratio = ratio 
        self.velocity = velocity 
        self.Tin = Tin
        self.model = model
        self.folder_name = folder_name

    def export_results(self):
        xo2_s = 2
        xn2_s = 2*0.79/0.21
        xch4_s = 1
        Mn2 = 28.0134
        Mo2 = 31.9988
        Mch4 = 16.043
        phi = self.ratio / 1000
        Mges = phi*xch4_s*Mch4+xo2_s*Mo2+xn2_s*Mn2
        ch4 = xch4_s*phi*Mch4/Mges
        o2 = xo2_s*Mo2/Mges
        N2 = xn2_s*Mn2/Mges
        self.N2 = N2
        self.o2 = o2
        self.ch4 = ch4
        
        export_dict = {}
        x = np.concatenate((np.ones([81, 65]) * 0, self.data_dict['CO']), axis = 1)
        export_dict['CO'] = np.where(x < 0, 0, x)
        x = np.concatenate((np.ones([81, 65]) * o2, self.data_dict['O2']), axis = 1)
        export_dict['O2'] = np.where(x < 0, 0, x)
        x = np.concatenate((np.ones([81, 65]) * 0, self.data_dict['CO2']), axis = 1)
        x = np.where(x >= o2, o2, x)
        export_dict['CO2'] = np.where(x < 0, 0, x)
        export_dict['Qdot'] = np.concatenate((np.zeros([81, 65]) * (self.velocity/100), self.data_dict['U1']), axis = 1)
        x = np.concatenate((np.ones([81, 65]) * ch4, self.data_dict['CH4']), axis = 1)
        x = np.where(x >= ch4, ch4, x)
        export_dict['CH4'] = np.where(x < 0, 0, x)
        x = np.concatenate((np.ones([81, 65]) * self.Tin, self.data_dict['T']), axis = 1)
        export_dict['T'] = np.where(x >= 300, x, self.Tin)
        x = np.concatenate((np.ones([81, 65]) * 0, self.data_dict['H2O']), axis = 1)
        export_dict['H2O'] = np.where(x < 0, 0, x)
        export_dict['U1'] = np.concatenate((np.ones([81, 65]) * (self.velocity/100), self.data_dict['U1']), axis = 1)
        export_dict['U3'] = np.concatenate((np.ones([81, 65]) *  0, self.data_dict['U3']), axis = 1)
        export_dict['U2'] = np.zeros([81, 306]) 
        export_dict['N2'] = np.ones([81, 306]) * N2
        self.export_dict = export_dict
        return export_dict


    def better_processing(self):
        vol = np.zeros((81, 306))
        gases = ['CH4','CO','H2O', 'N2']

        for i in gases:
            vol = vol + self.export_dict[i]
        export_dict_new = self.export_dict
        export_dict_new['O2'] = np.ones((81, 306)) - vol
        self.export_dict_new = export_dict_new
        return export_dict_new



    def pre_processing(self):
        gases = ['CH4','CO','O2','H2O', 'N2']
        a = np.zeros_like(self.export_dict['N2'])
        for i in gases:
            a = a + self.export_dict[i]

        rescaling_matrix = 1 / a

        export_dict_new = self.export_dict 
        b = np.zeros_like(self.export_dict['N2'])
        for i in gases:
            export_dict_new[i] = export_dict_new[i]  * rescaling_matrix
            b = b + export_dict_new[i] 

        self.export_dict_new = export_dict_new
        return export_dict_new

    def build_OF(self):
        src = 'OF_models/template_OF8'
        dst = 'OF_models/{}/{}/ER{}_Tin{}_Uin0{}_Twall373'.format(self.folder_name, self.model, self.ratio, self.Tin, self.velocity)
        shutil.copytree(src, dst)


        coords = np.genfromtxt('mesh/cell_dict.csv', delimiter=';')
        points = np.genfromtxt('mesh/points.csv', delimiter=';')
        points_2d = np.genfromtxt('mesh/points_2d.csv', delimiter=';')


        x = points_2d[:,0]
        y = points_2d[:,2]
        unique_x = np.sort(np.unique(x))
        unique_y = np.sort(np.unique(y))


        point_data = {}
        cell_data = {}

        channels = ['CO', 'O2', 'CO2', 'Qdot', 'CH4', 'T', 'H2O', 'N2', 'U1', 'U3']

        for i, channel_variable in enumerate(channels):
            
            export = self.export_dict_new[channel_variable]
            
            data_export = np.zeros(points.shape[0])
            data_generated = export
            for counter, row in enumerate(points):
                x_index = int(np.where(unique_x == row[0])[0])
                y_index = int(np.where(unique_y == row[2])[0])

                data_export[counter] = data_generated[y_index][x_index]
            point_data[channel_variable] = data_export

            
            cells = np.zeros(coords.shape[0])

            for i in range (coords.shape[0]):
                cells[i] = np.mean(point_data[channel_variable][coords[i].astype(int).tolist()])
            cell_data[channel_variable] = cells

        channels = ['CO', 'O2', 'CO2', 'Qdot', 'CH4', 'T', 'H2O', 'N2']
        path_to_folder = '{}/0/'.format(dst)

        for channel_variable in channels:
            data_export = cell_data[channel_variable]
            string = ''
            for i in range(data_export.shape[0]):
                string = string + str(data_export[i]) + '\n'
            
            if channel_variable == 'T':
                with open(path_to_folder + channel_variable, "r") as f:
                    contents = f.readlines()

                contents[31]='        value           uniform {};\n'.format(self.Tin)

                with open(path_to_folder + channel_variable, "w") as f:
                    contents = "".join(contents)
                    f.write(contents)


            if channel_variable == 'CH4':
                with open(path_to_folder + channel_variable, "r") as f:
                    contents = f.readlines()

                contents[31]='        value           uniform {};\n'.format(self.ch4)

                with open(path_to_folder + channel_variable, "w") as f:
                    contents = "".join(contents)
                    f.write(contents)


            if channel_variable == 'O2':
                with open(path_to_folder + channel_variable, "r") as f:
                    contents = f.readlines()

                contents[31]='        value           uniform {};\n'.format(self.o2)

                with open(path_to_folder + channel_variable, "w") as f:
                    contents = "".join(contents)
                    f.write(contents)


            if channel_variable == 'N2':
                with open(path_to_folder + channel_variable, "r") as f:
                    contents = f.readlines()

                contents[31]='        value           uniform {};\n'.format(self.N2)

                with open(path_to_folder + channel_variable, "w") as f:
                    contents = "".join(contents)
                    f.write(contents)

            
            with open(path_to_folder + channel_variable, "r") as f:
                contents = f.readlines()

            contents.insert(22, string)

            with open(path_to_folder + channel_variable, "w") as f:
                contents = "".join(contents)
                f.write(contents)

        u1 = cell_data['U1']
        u3 = cell_data['U3']

        string = ''
        for i in range(u1.shape[0]):
            string = string + '({} 0.0 {})'.format(u1[i], u3[i]) + '\n'

        with open(path_to_folder + 'U', "r") as f:
            contents = f.readlines()

        contents[31]='        value           uniform ({} 0 0);\n'.format(self.velocity/100)

        with open(path_to_folder + 'U', "w") as f:
            contents = "".join(contents)
            f.write(contents)

        with open(path_to_folder + 'U', "r") as f:
            contents = f.readlines()

        contents.insert(22, string)

        with open(path_to_folder + 'U', "w") as f:
            contents = "".join(contents)
            f.write(contents)

        
        


        


