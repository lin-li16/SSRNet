import torch.nn as nn


class CNN_allsites(nn.Module):
    def __init__(self, ker1=5, ker2=11, step=3, nums=256) -> None:
        super(CNN_allsites, self).__init__()
        self.step = step
        actfunc = nn.ReLU()
        self.encoder_eq = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Conv1d(16, 32, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Conv1d(32, 64, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Flatten(),
            nn.Linear(1408, 1024),
            actfunc,
            nn.Linear(1024, 512),
            actfunc,
            nn.Linear(512, nums),
            actfunc,
        )

        self.encoder_st = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Conv1d(8, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Conv1d(16, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Flatten(),
            nn.Linear(1600, 1024),
            actfunc,
            nn.Linear(1024, 512),
            actfunc,
            nn.Linear(512, nums),
            actfunc,
        )


        self.fc_eq_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(nums, nums),
            actfunc,
            nn.Linear(nums, nums),
            actfunc,
            nn.Linear(nums, nums),
            actfunc,
            ) for _ in range(step)
        ])

        actfunc1 = nn.Sigmoid()
        self.fc_st_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(nums, nums),
            actfunc1,
            nn.Linear(nums, nums),
            actfunc1,
            nn.Linear(nums, nums),
            actfunc1,
            ) for _ in range(step)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(nums, 128),
            # actfunc,
            nn.Linear(128, 128),
            # actfunc,
            nn.Linear(128, 66),
            # nn.ReLU(),
        )


    def forward(self, x):
        x0 = x[:, :132].reshape((x.shape[0], 6, 22))
        x1 = x[:, 132:].reshape((x.shape[0], 2, 100))
        x_eq = self.encoder_eq(x0)
        x_st = self.encoder_st(x1)
        for i in range(self.step):
            x_eq = self.fc_eq_list[i](x_eq)
            x_st = self.fc_st_list[i](x_st)
            x_eq = x_st * x_eq
        y = self.decoder(x_eq)
        return y


# class CNN_allsites(nn.Module):
#     def __init__(self, ker1=5, ker2=11, step=3, nums=256) -> None:
#         super(CNN_allsites, self).__init__()
#         self.step = step
#         actfunc = nn.ReLU()
#         self.encoder_eq = nn.Sequential(
#             nn.Conv1d(3, 16, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 32, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(32, 64, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1408, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )

#         self.encoder_st = nn.Sequential(
#             nn.Conv1d(2, 8, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(8, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1600, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )


#         self.fc_eq_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             ) for _ in range(step)
#         ])

#         actfunc1 = nn.Sigmoid()
#         self.fc_st_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             ) for _ in range(step)
#         ])

#         self.decoder = nn.Sequential(
#             nn.Linear(nums, 128),
#             # actfunc,
#             nn.Linear(128, 128),
#             # actfunc,
#             nn.Linear(128, 66),
#             # nn.ReLU(),
#         )


#     def forward(self, x):
#         x0 = x[:, :66].reshape((x.shape[0], 3, 22))
#         x1 = x[:, 132:].reshape((x.shape[0], 2, 100))
#         x_eq = self.encoder_eq(x0)
#         x_st = self.encoder_st(x1)
#         for i in range(self.step):
#             x_eq = self.fc_eq_list[i](x_eq)
#             x_st = self.fc_st_list[i](x_st)
#             x_eq = x_st * x_eq
#         y = self.decoder(x_eq)
#         return y
    

# class CNN_allsites(nn.Module):
#     def __init__(self, ker1=5, ker2=11, step=3, nums=256) -> None:
#         super(CNN_allsites, self).__init__()
#         self.step = step
#         actfunc = nn.ReLU()
#         self.encoder_eq = nn.Sequential(
#             nn.Conv1d(6, 16, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 32, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(32, 64, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1408, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )

#         self.encoder_st = nn.Sequential(
#             nn.Conv1d(2, 8, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(8, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1600, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )


#         self.fc_eq_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             ) for _ in range(step)
#         ])

#         actfunc1 = nn.Sigmoid()
#         self.fc_st_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             ) for _ in range(step)
#         ])

#         self.decoder = nn.Sequential(
#             nn.Linear(nums, 128),
#             # actfunc,
#             nn.Linear(128, 128),
#             # actfunc,
#             nn.Linear(128, 66),
#             # nn.ReLU(),
#         )


    # def forward(self, x):
    #     x0 = x[:, :132].reshape((x.shape[0], 6, 22))
    #     # x1 = x[:, 132:].reshape((x.shape[0], 2, 100))
    #     x_eq = self.encoder_eq(x0)
    #     # x_st = self.encoder_st(x1)
    #     for i in range(self.step):
    #         x_eq = self.fc_eq_list[i](x_eq)
    #         # x_st = self.fc_st_list[i](x_st)
    #         # x_eq = x_st * x_eq
    #     y = self.decoder(x_eq)
    #     return y



# class CNN_allsites(nn.Module):
#     def __init__(self, ker1=5, ker2=11, step=3, nums=256) -> None:
#         super(CNN_allsites, self).__init__()
#         self.step = step
#         actfunc = nn.ReLU()
#         self.encoder_eq = nn.Sequential(
#             nn.Conv1d(3, 16, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 32, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(32, 64, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1408, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )

#         self.encoder_st = nn.Sequential(
#             nn.Conv1d(2, 8, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(8, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Conv1d(16, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
#             actfunc,
#             nn.Flatten(),
#             nn.Linear(1600, 1024),
#             actfunc,
#             nn.Linear(1024, 512),
#             actfunc,
#             nn.Linear(512, nums),
#             actfunc,
#         )


#         self.fc_eq_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             nn.Linear(nums, nums),
#             actfunc,
#             ) for _ in range(step)
#         ])

#         actfunc1 = nn.Sigmoid()
#         self.fc_st_list = nn.ModuleList([
#             nn.Sequential(
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             nn.Linear(nums, nums),
#             actfunc1,
#             ) for _ in range(step)
#         ])

#         self.decoder = nn.Sequential(
#             nn.Linear(nums, 128),
#             # actfunc,
#             nn.Linear(128, 128),
#             # actfunc,
#             nn.Linear(128, 66),
#             # nn.ReLU(),
#         )


#     def forward(self, x):
#         x0 = x[:, :66].reshape((x.shape[0], 3, 22))
#         # x1 = x[:, 132:].reshape((x.shape[0], 2, 100))
#         x_eq = self.encoder_eq(x0)
#         # x_st = self.encoder_st(x1)
#         for i in range(self.step):
#             x_eq = self.fc_eq_list[i](x_eq)
#             # x_st = self.fc_st_list[i](x_st)
#             # x_eq = x_st * x_eq
#         y = self.decoder(x_eq)
#         return y
    


class CNN_retrain(nn.Module):
    def __init__(self, ker1=5, ker2=11, step=3) -> None:
        super(CNN_retrain, self).__init__()
        self.step = step
        actfunc = nn.ReLU()
        self.encoder_eq = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Conv1d(16, 32, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Conv1d(32, 64, kernel_size=ker1, padding=int((ker1 - 1) / 2)),
            actfunc,
            nn.Flatten(),
            nn.Linear(1408, 1024),
            actfunc,
            nn.Linear(1024, 512),
            actfunc,
            nn.Linear(512, 256),
            actfunc,
        )

        self.encoder_st = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Conv1d(8, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Conv1d(16, 16, kernel_size=ker2, padding=int((ker2 - 1) / 2)),
            actfunc,
            nn.Flatten(),
            nn.Linear(1600, 1024),
            actfunc,
            nn.Linear(1024, 512),
            actfunc,
            nn.Linear(512, 256),
            actfunc,
        )

        nums = 256
        self.fc_eq_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(nums, nums),
            actfunc,
            nn.Linear(nums, nums),
            actfunc,
            nn.Linear(nums, nums),
            actfunc,
            ) for _ in range(step)
        ])

        self.fc_st_list = nn.ModuleList([
            nn.Sequential(
            nn.Linear(nums, nums),
            nn.Sigmoid(),
            nn.Linear(nums, nums),
            nn.Sigmoid(),
            nn.Linear(nums, nums),
            nn.Sigmoid(),
            ) for _ in range(step)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(nums, 128),
            # actfunc,
            nn.Linear(128, 128),
            # actfunc,
            nn.Linear(128, 66),
            # nn.ReLU(),
        )

        self.retrain = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 66),
            # nn.ReLU(),
        )


    def forward(self, x):
        x0 = x[:, :154].reshape((x.shape[0], 7, 22))
        x1 = x[:, 154:].reshape((x.shape[0], 2, 100))
        x_eq = self.encoder_eq(x0)
        x_st = self.encoder_st(x1)
        for i in range(self.step):
            x_eq = self.fc_eq_list[i](x_eq)
            x_st = self.fc_st_list[i](x_st)
            x_eq = x_st * x_eq
        y = self.decoder(x_eq)
        y = self.retrain(y)
        return y