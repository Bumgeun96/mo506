from dqn import DQN
from loading_bdd100k import get_image_center
from image_captioning import image_to_text

class Player():
    def __init__(self):
        self.dqn_obs_size = 6 #(x, y, x_a, y_a, h, v)
        self.dqn_act_size = 4 # number of discrete actions
        self.max_episode = 100
        self.max_epoch = 1000
        self.dqn_trainer = DQN(self.dqn_obs_size, self.dqn_act_size)
        self.image_to_text = image_to_text()
        self.done = False
        
    def image_truncation(self,images,center = [50,50],hor = 10,ver = 10):
        croppedImages = []
        for image in images:
            horizontal,vertical = image.size
            croppedImage = image.crop((center[0]-hor,center[1]-ver,center[0]+hor,center[1]+ver))
            croppedImages.append(croppedImage)
        return croppedImages
        
    def reset(self):
        image,center,center_attention = get_image_center(n=1)
        state = [center[0][0],center[0][1],center_attention[0][0],center_attention[0][1],10,10]
        self.done = False
        return state,image
    
    def step(self,state,action,image):
        horizontal,vertical = image[0].size
        center_x = state[0]
        center_y = state[1]
        if action == 0: # +1 horizontal
            if center_x + state[-2] > horizontal-2:
                state [-2] = state[-2]
            else:
                state[-2] = state[-2] + 1
        elif action == 1: # -1 horizontal
            if state[-2] < 11:
                state[-2] = state[-2]
            else:
                state[-2] = state[-2] - 1
        elif action == 2: # +1 vertical
            if center_y + state[-1] > vertical-2:
                state[-1] = state[-1]
            else:
                state[-1] = state[-1] + 1
        elif action == 3: # -1 vertical
            if state[-1] < 11:
                state[-1] = state[-1]
            else:
                state[-1] = state[-1] - 1
        next_state = state
        return next_state
    
    def get_reward(self,state,next_state,image):
        current_image = self.image_truncation(image,[state[0],state[1]],hor=state[-2],ver=state[-1])
        next_image = self.image_truncation(image,center = [next_state[0],next_state[1]],hor=next_state[-2],ver=next_state[-1])
        current_text = self.image_to_text.image_captioning(current_image[0])
        next_text = self.image_to_text.image_captioning(next_image[0])
        print(current_text[0]['generated_text'])
        # self.image_to_text.score()
        # next_blew_score = model(next_image)
        reward = 1
        return reward

    def run(self):
        for __ in range(self.max_episode):
            print("epi:",__+1)
            self.dqn_trainer.Epsilon()
            self.dqn_trainer.Loss()
            state, image = self.reset()
            for _ in range(self.max_epoch):
                action = self.dqn_trainer.select_action(state)
                # print(state[-2:])
                next_state = self.step(state,action,image)
                reward = self.get_reward(state,next_state,image)
                if _ == self.max_epoch-1:
                    self.done = True
                self.dqn_trainer.store_experience(state,next_state,action,reward,self.done)
                state = next_state
                self.dqn_trainer.update()


if __name__ == '__main__':
    player = Player()
    player.run()