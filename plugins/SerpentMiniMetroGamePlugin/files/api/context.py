

class Context(object):

    @staticmethod
    def click(x, y, new_context, ctrl):
        ctrl.move(x=x, y=y)
        ctrl.click()
        return new_context


class MainMenu(Context):

    @classmethod
    def click_play(cls, ctrl):
        return cls.click(x=270, y=210, new_context=LevelMenu, ctrl=ctrl)


class LevelMenu(Context):

    @classmethod
    def click_play(cls, ctrl):
        return cls.click(x=750, y=500, new_context=GamePlay, ctrl=ctrl)

    @classmethod
    def click_mode(cls, ctrl):
        return cls.click(x=750, y=540, new_context=ModeMenu, ctrl=ctrl)


class ModeMenu(Context):

    @classmethod
    def click_normal(cls, ctrl):
        return cls.click(x=400, y=180, new_context=cls, ctrl=ctrl)

    @classmethod
    def click_extreme(cls, ctrl):
        return cls.click(x=400, y=220, new_context=cls, ctrl=ctrl)

    @classmethod
    def click_endless(cls, ctrl):
        return cls.click(x=400, y=260, new_context=cls, ctrl=ctrl)

    @classmethod
    def click_creative(cls, ctrl):
        return cls.click(x=400, y=300, new_context=cls, ctrl=ctrl)

    @classmethod
    def click_return(cls, ctrl):
        return cls.click(x=30, y=20, new_context=LevelMenu, ctrl=ctrl)


class GamePlay(Context):
    pass
