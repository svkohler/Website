from doctest import register_optionflag
from unittest.mock import _SentinelObject
import ee
import geemap
import os
import numpy as np
from PIL import Image
from skimage import io
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


def get_sat_image(coordinates, PATH):
    ee.Initialize()

    bottom_left = ee.Geometry.Point(
        [coordinates['bottom_left_long'], coordinates['bottom_left_lat']])
    top_right = ee.Geometry.Point(
        [coordinates['top_right_long'], coordinates['top_right_lat']])

    area = ee.Geometry.Rectangle([bottom_left, top_right])

    sentinel = ee.ImageCollection(
        'COPERNICUS/S2_SR').filterBounds(area).filterDate('2018-01-01', '2018-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)).map(maskS2clouds)

    sentinel_SR_RGB = sentinel.select(['B4', 'B3', 'B2']).mosaic(
    ).visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.3)

    geemap.ee_export_image(
        sentinel_SR_RGB, filename=PATH+'satellite.tif', scale=10, region=area, file_per_band=False)

    sentinel_rgb = Image.fromarray(
        np.uint8(io.imread(PATH+'satellite.tif')))

    sentinel_rgb = sentinel_rgb.resize((224, 224))

    sentinel_rgb.save(PATH+'satellite.png', format="png")

    os.remove(PATH+'satellite.tif')


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) and (
        qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return out


class MLP_classifier(nn.Module):
    def __init__(self, device):
        super(MLP_classifier, self).__init__()
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.num_epochs = 200
        self.device = device

    def forward(self, x):
        out = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return out

    def fit(self, X, y):
        self.model = MLP(X.shape[1], 100, 1)
        self.model.to(self.device)

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1)

        training_samples = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train))
        validation_samples = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val))
        data_loader_trn = torch.utils.data.DataLoader(
            training_samples, batch_size=200, drop_last=False, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(
            validation_samples, batch_size=200, drop_last=False, shuffle=True)

        val_loss = None
        counter = 0
        for epoch in range(self.num_epochs):
            for batch_idx, (data, target) in enumerate(data_loader_trn):

                tr_x, tr_y = data.to(self.device), target.to(self.device)

                pred = self.model(tr_x)
                loss = self.criterion(torch.squeeze(pred), tr_y.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch > 0:
                with torch.no_grad():
                    current_validation_loss = 0
                    for batch_idx, (data, target) in enumerate(data_loader_val):
                        val_x, val_y = data.float(), target.float()
                        val_x, val_y = data.to(
                            self.device), target.to(self.device)
                        pred = self.model(val_x)
                        current_validation_loss += self.criterion(
                            torch.squeeze(pred), val_y.float())
                    # check for early stopping
                    if val_loss is None:
                        val_loss = current_validation_loss
                    elif val_loss < current_validation_loss + 0.0001:
                        counter += 1
                        if counter >= 10:
                            print(f'early stopping after {epoch} epochs.')
                            break
                    elif val_loss > current_validation_loss + 0.0001:
                        val_loss = current_validation_loss
                        counter = 0

    def predict(self, X_test):
        with torch.no_grad():
            x = torch.from_numpy(X_test)
            x = x.to(self.device)
            # labels = torch.from_numpy(y_test).type(torch.float)
            outputs = torch.squeeze(self.model(
                x)).cpu().round().detach().numpy()
            return outputs

    def save_model(self, config):
        torch.save(self.model.state_dict(), config.dump_path +
                   '/clf/'+'_'+config.model_name+str(config.patch_size)+'.pth')


class TrueForestDataset(Dataset):
    '''
    dataset to train TrueForest models
    '''

    def __init__(self, config, mode, transform=True):
        self.config = config
        self.mode = mode
        self.trans = transform
        self.satellite_rgb_dir = self.config.data_store + '/satellite_rgb/' + \
            config.location + '/' + self.mode + \
            '/' + str(config.patch_size) + '/'
        # self.satellite_nir_dir = self.config.data_store + '/satellite_nir/' + \
        #     config.location + '/' + self.mode + \
        #     '/' + str(config.patch_size) + '/'
        self.drone_dir = self.config.data_store + '/drone/' + \
            config.location + '/' + self.mode + \
            '/' + str(config.patch_size) + '/'
        self.len = self.check_len()
        self.satellite_rgb_images = sorted(os.listdir(self.satellite_rgb_dir))
        # self.satellite_nir_images = sorted(os.listdir(self.satellite_nir_dir))
        self.drone_images = sorted(os.listdir(self.drone_dir))

        # transformation parameter
        self.angles = [0, 90, 180, 270]
        self.contrast_range = [0.6, 2]
        self.gamma_range = [0.8, 1.3]
        self.hue_range = [-0.3, 0.4]
        self.saturation_range = [0, 2]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_satellite = ToTensor()(Image.open(
            self.satellite_rgb_dir + self.satellite_rgb_images[idx]))
        # augment with near infrared channel if activated
        # if self.config.NIR:
        #     img_satellite_nir = ToTensor()(Image.open(
        #         self.satellite_nir_dir + self.satellite_nir_images[idx]))
        #     img_satellite = torch.cat(
        #         (img_satellite, img_satellite_nir), dim=0)
        img_drone = ToTensor()(Image.open(
            self.drone_dir + self.drone_images[idx]))
        # perform transformations
        if self.trans:
            img_satellite, img_drone = self.transform(img_satellite, img_drone)

        return img_satellite, img_drone

    def transform(self, satellite, drone):

        if self.config.transforms.hflip and torch.rand(1) < self.config.transforms.hflip_prob:
            satellite = transforms.functional.hflip(satellite)
            drone = transforms.functional.hflip(drone)

        if self.config.transforms.vflip and torch.rand(1) < self.config.transforms.vflip_prob:
            satellite = transforms.functional.vflip(satellite)
            drone = transforms.functional.vflip(drone)

        if self.config.transforms.gaussian_blur and torch.rand(1) < self.config.transforms.gaussian_blur_prob:
            blurrer = transforms.GaussianBlur(
                kernel_size=[23, 23], sigma=(0.1, 2.0))
            satellite = blurrer(satellite)
            drone = blurrer(drone)

        if self.config.transforms.contrast and torch.rand(1) < self.config.transforms.contrast_prob:
            contrast = self.get_param(torch.rand(1), self.contrast_range)
            satellite = transforms.functional.adjust_contrast(
                satellite, contrast)
            drone = transforms.functional.adjust_contrast(drone, contrast)

        if self.config.transforms.hue and torch.rand(1) < self.config.transforms.hue_prob:
            gamma = self.get_param(torch.rand(1), self.gamma_range)
            satellite = transforms.functional.adjust_gamma(satellite, gamma)
            drone = transforms.functional.adjust_gamma(drone, gamma)

        if self.config.transforms.gamma and torch.rand(1) < self.config.transforms.gamma_prob:
            hue = self.get_param(torch.rand(1), self.hue_range)
            satellite = transforms.functional.adjust_hue(satellite, hue)
            drone = transforms.functional.adjust_hue(drone, hue)

        if self.config.transforms.saturation and torch.rand(1) < self.config.transforms.saturation_prob:
            saturation = self.get_param(torch.rand(1), self.saturation_range)
            satellite = transforms.functional.adjust_saturation(
                satellite, saturation)
            drone = transforms.functional.adjust_saturation(drone, saturation)

        if self.config.transforms.rotate:
            idx = int(torch.floor(torch.rand(1)*4))
            satellite = transforms.functional.rotate(
                satellite, self.angles[idx])
            drone = transforms.functional.rotate(drone, self.angles[idx])

        if self.config.transforms.normalize:
            satellite = transforms.functional.normalize(
                satellite, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            drone = transforms.functional.normalize(
                drone, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return satellite, drone

    def check_len(self):
        if len(os.listdir(self.satellite_rgb_dir)) == len(os.listdir(self.drone_dir)):
            return len(os.listdir(self.satellite_rgb_dir))
        else:
            raise ValueError(
                'There is not the same number of drone and satellite images.')

    def get_param(self, random_nr, range):
        '''
        helper function to get transform parameter in the correct range
        '''
        return range[0] + (range[1]-range[0])*random_nr
