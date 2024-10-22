import torch
from torch.utils.hooks import RemovableHandle
from functools import partial
import torch.nn as nn
import traceback


class InspectOutputContext:
    def __init__(self, model, module_names, move_to_cpu=False, last_position=False):
        self.model = model
        self.module_names = module_names
        self.move_to_cpu = move_to_cpu
        self.last_position = last_position
        self.handles = []
        self.catcher = dict()

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.module_names:
                handle = inspect_output(module, self.catcher, module_name, move_to_cpu=self.move_to_cpu,
                                        last_position=self.last_position)
                self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True


class ReplaceOutputContext:
    def __init__(self, model, module_name, x, start_end_ids):
        self.model = model
        self.module_name = module_name
        self.start_end_ids = start_end_ids
        self.replace_handle = None
        self.x = x

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name == self.module_name:
                self.replace_handle = replace_output_with_x(module, self.x, self.start_end_ids)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.replace_handle.remove()
        # self.inspect_handle.remove()
        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True


class PatchOutputContext:
    def __init__(self, model, module_name, func, position):
        self.model = model
        self.module_name = module_name
        self.position = position
        self.patch_handle = None
        self.func = func

    def __enter__(self):
        if type(self.module_name) is str:
            for module_name, module in self.model.named_modules():
                if module_name == self.module_name:
                    self.patch_handle = patch_output(module, self.func, self.position)
        elif type(self.module_name) is list:
            self.patch_handle = []
            assert len(self.func) == len(self.module_name)
            for cur_module, cur_func in zip(self.module_name, self.func):
                for module_name, module in self.model.named_modules():
                    if module_name == cur_module:
                        cur_patch_handle = patch_output(module, cur_func, self.position)
                        self.patch_handle.append(cur_patch_handle)
        else:
            raise ValueError
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if type(self.patch_handle) is list:
            for patch_handle in self.patch_handle:
                patch_handle.remove()
        else:
            self.patch_handle.remove()
        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True


def patch_hook(module: nn.Module, inputs, outputs, func, position):
    if type(outputs) is tuple:
        # outputs[0]: [batch_size, seq_length, hidden_size]
        outputs[0][:, position] = func(outputs[0][:, position])
        outputs[0].contiguous()
    else:
        outputs[:, position] = func(outputs[:, position])
        outputs.contiguous()
    return outputs


def replace_hook(module: nn.Module, inputs, outputs, x, start_end_ids):
    if type(outputs) is tuple:
        # outputs[0]: [batch_size, seq_length, hidden_size]
        for s, e in start_end_ids:
            outputs[0][:, s:e] = x[:, s:e].to(outputs[0].device)
        outputs[0].contiguous()
    else:
        for s, e in start_end_ids:
            outputs[:, s:e] = x[:, s:e].to(outputs.device)
        outputs.contiguous()
    return outputs


def inspect_hook(module: nn.Module, inputs, outputs, catcher: dict, module_name, move_to_cpu, last_position=False):
    if last_position:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0][:, -1]  # .clone()
        else:
            catcher[module_name] = outputs[:, -1]
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    else:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0]  # .clone()
        else:
            catcher[module_name] = outputs
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    return outputs


def patch_output(module: nn.Module, func: callable, position) -> RemovableHandle:
    hook_instance = partial(patch_hook, func=func, position=position)
    handle = module.register_forward_hook(hook_instance)
    return handle


def replace_output_with_x(module: nn.Module, x: torch.Tensor, start_end_ids) -> RemovableHandle:
    hook_instance = partial(replace_hook, x=x, start_end_ids=start_end_ids)
    handle = module.register_forward_hook(hook_instance)
    return handle


def inspect_output(module: nn.Module, catcher: dict, module_name, move_to_cpu, last_position=False) -> RemovableHandle:
    hook_instance = partial(inspect_hook, catcher=catcher, module_name=module_name, move_to_cpu=move_to_cpu,
                            last_position=last_position)
    handle = module.register_forward_hook(hook_instance)
    return handle
